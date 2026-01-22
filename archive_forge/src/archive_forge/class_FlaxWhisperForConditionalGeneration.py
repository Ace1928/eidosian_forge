import math
import random
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
@add_start_docstrings('The Whisper Model with a language modeling head.', WHISPER_START_DOCSTRING)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, past_key_values: dict=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        """
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        encoder_hidden_states = encoder_outputs[0]
        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `decoder_position_ids` when passing `past_key_values`.')
            if decoder_attention_mask is not None:
                decoder_position_ids = decoder_attention_mask.cumsum(-1) * decoder_attention_mask - 1
            else:
                decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length), dtype='i4')
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, **kwargs)
            hidden_states = outputs[0]
            if self.config.tie_word_embeddings:
                shared_embedding = module.model.decoder.embed_tokens.variables['params']['embedding']
                lm_logits = module.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
            else:
                lm_logits = module.lm_head(hidden_states)
            return (lm_logits, outputs)
        outputs = self.module.apply(inputs, decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), encoder_hidden_states=encoder_hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, mutable=mutable, method=_decoder_forward)
        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs
        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(logits=lm_logits, hidden_states=decoder_outputs.hidden_states, attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions)
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]
        if past_key_values is not None and return_dict:
            outputs['past_key_values'] = unfreeze(past['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            outputs = outputs[:1] + (unfreeze(past['cache']),) + outputs[1:]
        return outputs

    def generate(self, input_features, generation_config=None, logits_processor=None, return_timestamps=None, task=None, language=None, is_multilingual=None, **kwargs):
        if generation_config is None:
            generation_config = self.generation_config
        if return_timestamps is not None:
            generation_config.return_timestamps = return_timestamps
        if task is not None:
            generation_config.task = task
        if is_multilingual is not None:
            generation_config.is_multilingual = is_multilingual
        if language is not None:
            generation_config.language = language
        if kwargs is not None and 'decoder_input_ids' in kwargs:
            decoder_input_length = len(kwargs['decoder_input_ids'])
        else:
            decoder_input_length = 1
        forced_decoder_ids = []
        if hasattr(generation_config, 'is_multilingual') and generation_config.is_multilingual:
            if hasattr(generation_config, 'language'):
                forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
            else:
                forced_decoder_ids.append((1, None))
            if hasattr(generation_config, 'task'):
                forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id['transcribe']))
        if hasattr(generation_config, 'return_timestamps') and generation_config.return_timestamps or return_timestamps:
            logits_processor = [FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)]
        elif forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
            idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
            forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))
        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids
        return super().generate(input_features, generation_config, logits_processor=logits_processor, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array]=None, decoder_attention_mask: Optional[jax.Array]=None, encoder_outputs=None, **kwargs):
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype='i4')[None, :], (batch_size, seq_length))
        return {'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'encoder_attention_mask': attention_mask, 'decoder_attention_mask': extended_attention_mask, 'decoder_position_ids': position_ids}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['decoder_position_ids'] = model_kwargs['decoder_position_ids'][:, -1:] + 1
        return model_kwargs