from typing import Dict, Optional, Tuple
import flax
import jax.numpy as jnp
from .utils import ModelOutput
@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """
    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None