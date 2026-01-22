import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
from .stopping_criteria import (
def _constrained_beam_search(self, input_ids: torch.LongTensor, constrained_beam_scorer: ConstrainedBeamSearchScorer, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, output_logits: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: Optional[bool]=None, **model_kwargs) -> Union[GenerateBeamOutput, torch.LongTensor]:
    """
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._constrained_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     ConstrainedBeamSearchScorer,
        ...     PhrasalConstraint,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> constraint_str = "Sie"
        >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
        >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


        >>> # instantiate beam scorer
        >>> beam_scorer = ConstrainedBeamSearchScorer(
        ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model._constrained_beam_search(
        ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt sind Sie?']
        ```"""
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn('`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.', UserWarning)
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    if eos_token_id is not None:
        logger.warning_once('`eos_token_id` is deprecated in this function and will be removed in v4.41, use `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead. Otherwise make sure to set `model.generation_config.eos_token_id`', FutureWarning)
        stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    else:
        eos_token_id = [criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, 'eos_token_id')]
        eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None and self.generation_config.eos_token_id is not None:
            eos_token_id = self.generation_config.eos_token_id
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
    batch_size = len(constrained_beam_scorer._beam_hyps)
    num_beams = constrained_beam_scorer.num_beams
    batch_beam_size, cur_len = input_ids.shape
    if 'inputs_embeds' in model_kwargs:
        cur_len = model_kwargs['inputs_embeds'].shape[1]
    model_kwargs['cache_position'] = torch.arange(cur_len, device=input_ids.device)
    if num_beams * batch_size != batch_beam_size:
        raise ValueError(f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.')
    scores = () if return_dict_in_generate and output_scores else None
    raw_logits = () if return_dict_in_generate and output_logits else None
    beam_indices = tuple((() for _ in range(batch_beam_size))) if return_dict_in_generate and output_scores else None
    decoder_attentions = () if return_dict_in_generate and output_attentions else None
    cross_attentions = () if return_dict_in_generate and output_attentions else None
    decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
        encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1000000000.0
    beam_scores = beam_scores.view((batch_size * num_beams,))
    this_peer_finished = False
    decoder_prompt_len = input_ids.shape[-1]
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)
        scores_for_all_vocab = next_token_scores.clone()
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True)
        next_indices = (next_tokens / vocab_size).long()
        next_tokens = next_tokens % vocab_size
        beam_outputs = constrained_beam_scorer.process(input_ids, next_token_scores, next_tokens, next_indices, scores_for_all_vocab, pad_token_id=pad_token_id, eos_token_id=eos_token_id, beam_indices=beam_indices, decoder_prompt_len=decoder_prompt_len)
        beam_scores = beam_outputs['next_beam_scores']
        beam_next_tokens = beam_outputs['next_beam_tokens']
        beam_idx = beam_outputs['next_beam_indices']
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        if model_kwargs.get('past_key_values', None) is not None:
            model_kwargs['past_key_values'] = self._temporary_reorder_cache(model_kwargs['past_key_values'], beam_idx)
        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
        cur_len = cur_len + 1
        if constrained_beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True
    sequence_outputs = constrained_beam_scorer.finalize(input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id, max_length=stopping_criteria.max_length, beam_indices=beam_indices, decoder_prompt_len=decoder_prompt_len)
    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs['sequence_scores'] = None
        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(sequences=sequence_outputs['sequences'], sequences_scores=sequence_outputs['sequence_scores'], scores=scores, logits=raw_logits, beam_indices=sequence_outputs['beam_indices'], encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
        else:
            return GenerateBeamDecoderOnlyOutput(sequences=sequence_outputs['sequences'], sequences_scores=sequence_outputs['sequence_scores'], scores=scores, logits=raw_logits, beam_indices=sequence_outputs['beam_indices'], attentions=decoder_attentions, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
    else:
        return sequence_outputs['sequences']