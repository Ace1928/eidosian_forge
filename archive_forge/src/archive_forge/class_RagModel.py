import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):

    def __init__(self, config: Optional[PretrainedConfig]=None, question_encoder: Optional[PreTrainedModel]=None, generator: Optional[PreTrainedModel]=None, retriever: Optional[RagRetriever]=None, **kwargs):
        assert config is not None or (question_encoder is not None and generator is not None), 'Either a configuration or an question_encoder and a generator has to be provided.'
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        else:
            assert isinstance(config, self.config_class), f'config: {config} has to be of type {self.config_class}'
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel
            question_encoder = AutoModel.from_config(config.question_encoder)
        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = AutoModelForSeq2SeqLM.from_config(config.generator)
        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(retriever, RagRetriever), f'`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`'
            self.retriever = retriever
        self.question_encoder = question_encoder
        self.generator = generator
        self.ctx_encoder = None
        self.context_encoder_training = False

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, doc_scores: Optional[torch.FloatTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, n_docs: Optional[int]=None) -> Union[Tuple[torch.Tensor], RetrievAugLMOutput]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> outputs = model(input_ids=inputs["input_ids"])
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved
        has_to_retrieve = self.retriever is not None and (context_input_ids is None or context_attention_mask is None or doc_scores is None) and (encoder_outputs is None)
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
                question_encoder_last_hidden_state = question_enc_outputs[0]
                retriever_outputs = self.retriever(input_ids, question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')
                if self.context_encoder_training:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrived_doc_input_ids, retrived_doc_attention_mask, retrieved_doc_ids = (retriever_outputs['context_input_ids'], retriever_outputs['context_attention_mask'], retriever_outputs['retrieved_doc_embeds'], retriever_outputs['tokenized_doc_ids'], retriever_outputs['tokenized_doc_attention_mask'], retriever_outputs['doc_ids'])
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)
                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(-1, n_docs, question_encoder_last_hidden_state.shape[1])
                    doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (retriever_outputs['context_input_ids'], retriever_outputs['context_attention_mask'], retriever_outputs['retrieved_doc_embeds'], retriever_outputs['doc_ids'])
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)
                    doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            else:
                assert context_input_ids is not None, 'Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert context_attention_mask is not None, 'Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert doc_scores is not None, 'Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
        assert doc_scores is not None, 'Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function.'
        assert doc_scores.shape[1] % n_docs == 0, f' The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}.'
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)
        gen_outputs = self.generator(input_ids=context_input_ids, attention_mask=context_attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, return_dict=True)
        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions
        if not has_to_retrieve or not output_retrieved:
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        return RetrievAugLMOutput(logits=gen_outputs.logits, doc_scores=doc_scores, past_key_values=gen_outputs.past_key_values, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, retrieved_doc_embeds=retrieved_doc_embeds, retrieved_doc_ids=retrieved_doc_ids, question_encoder_last_hidden_state=question_encoder_last_hidden_state, question_enc_hidden_states=question_enc_hidden_states, question_enc_attentions=question_enc_attentions, generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state, generator_enc_hidden_states=gen_outputs.encoder_hidden_states, generator_enc_attentions=gen_outputs.encoder_attentions, generator_dec_hidden_states=gen_outputs.decoder_hidden_states, generator_dec_attentions=gen_outputs.decoder_attentions, generator_cross_attentions=gen_outputs.cross_attentions)