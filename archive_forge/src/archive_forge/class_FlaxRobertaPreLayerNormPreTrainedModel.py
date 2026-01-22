from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig
class FlaxRobertaPreLayerNormPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RobertaPreLayerNormConfig
    base_model_prefix = 'roberta_prelayernorm'
    module_class: nn.Module = None

    def __init__(self, config: RobertaPreLayerNormConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, gradient_checkpointing: bool=False, **kwargs):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype='i4')
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states, encoder_attention_mask, return_dict=False)
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False)
        random_params = module_init_outputs['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        """
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids, dtype='i4')
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, past_key_values: dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if self.config.add_cross_attention:
            if past_key_values:
                inputs['cache'] = past_key_values
                mutable = ['cache']
            else:
                mutable = False
            outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), token_type_ids=jnp.array(token_type_ids, dtype='i4'), position_ids=jnp.array(position_ids, dtype='i4'), head_mask=jnp.array(head_mask, dtype='i4'), encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, deterministic=not train, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, rngs=rngs, mutable=mutable)
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs['past_key_values'] = unfreeze(past_key_values['cache'])
                return outputs
            elif past_key_values is not None and (not return_dict):
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values['cache']),) + outputs[1:]
        else:
            outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), token_type_ids=jnp.array(token_type_ids, dtype='i4'), position_ids=jnp.array(position_ids, dtype='i4'), head_mask=jnp.array(head_mask, dtype='i4'), deterministic=not train, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, rngs=rngs)
        return outputs