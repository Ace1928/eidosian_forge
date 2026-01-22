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
def _initialize_attention(model_kwargs, num_padding_values, is_encoder_decoder):
    """initializes the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
    if is_encoder_decoder:
        decoder_attention_mask = tf.concat([tf.ones((batch_size, 1), dtype=tf.int32), tf.zeros((batch_size, num_padding_values), dtype=tf.int32), tf.ones((batch_size, 1), dtype=tf.int32)], axis=1)
        mask = {'decoder_attention_mask': decoder_attention_mask}
    else:
        attention_mask = model_kwargs.pop('attention_mask')
        attention_mask = tf.concat([attention_mask, tf.zeros((batch_size, num_padding_values), dtype=attention_mask.dtype), tf.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=1)
        mask = {'attention_mask': attention_mask}
    return mask