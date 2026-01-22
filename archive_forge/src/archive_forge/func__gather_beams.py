import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
@staticmethod
def _gather_beams(nested, beam_indices, batch_axis=0):
    """Gathers the beam slices indexed by beam_indices into new beam array."""

    def gather_fn(tensor):
        if batch_axis > 0:
            perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
            tensor = tf.transpose(tensor, perm=perm)
        gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
        if batch_axis > 0:
            perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
            perm = tf.math.invert_permutation(perm)
            gathered_tensor = tf.transpose(gathered_tensor, perm=perm)
        return gathered_tensor
    return tf.nest.map_structure(gather_fn, nested)