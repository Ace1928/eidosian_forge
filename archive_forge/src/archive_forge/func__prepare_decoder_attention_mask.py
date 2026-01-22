from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
    batch_size, seq_len = (input_shape[0], input_shape[1])
    combined_attention_mask = tf.cond(tf.math.greater(seq_len, 1), lambda: _make_causal_mask(input_shape, past_key_values_length=past_key_values_length), lambda: _expand_mask(tf.ones((batch_size, seq_len + past_key_values_length)), tgt_len=seq_len))
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    return combined_attention_mask