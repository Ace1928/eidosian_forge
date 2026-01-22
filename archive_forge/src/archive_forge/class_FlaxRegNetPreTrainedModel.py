from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
from transformers.modeling_flax_utils import (
from transformers.utils import (
class FlaxRegNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RegNetConfig
    base_model_prefix = 'regnet'
    main_input_name = 'pixel_values'
    module_class: nn.Module = None

    def __init__(self, config: RegNetConfig, input_shape=(1, 224, 224, 3), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)
        rngs = {'params': rng}
        random_params = self.module.init(rngs, pixel_values, return_dict=False)
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    def __call__(self, pixel_values, params: dict=None, train: bool=False, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        rngs = {}
        return self.module.apply({'params': params['params'] if params is not None else self.params['params'], 'batch_stats': params['batch_stats'] if params is not None else self.params['batch_stats']}, jnp.array(pixel_values, dtype=jnp.float32), not train, output_hidden_states, return_dict, rngs=rngs, mutable=['batch_stats'] if train else False)