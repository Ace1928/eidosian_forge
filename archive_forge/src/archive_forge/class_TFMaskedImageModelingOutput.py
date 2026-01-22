from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
import tensorflow as tf
from .utils import ModelOutput
@dataclass
class TFMaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss.
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: tf.Tensor | None = None
    reconstruction: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

    @property
    def logits(self):
        warnings.warn('logits attribute is deprecated and will be removed in version 5 of Transformers. Please use the reconstruction attribute to retrieve the final output instead.', FutureWarning)
        return self.reconstruction