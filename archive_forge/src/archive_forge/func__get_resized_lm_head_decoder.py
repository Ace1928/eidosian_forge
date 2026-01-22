from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
    """
        Build a resized decoder from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_decoder (`tf.Variable`):
                Old lm head decoder to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized decoder or None if the output embeddings are different from the input
            ones.
        """
    new_lm_head_decoder = old_lm_head_decoder
    is_input_output_equals = tf.reduce_any(self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder)
    if old_lm_head_decoder is not None and (not is_input_output_equals):
        old_embedding_dim = shape_list(old_lm_head_decoder)[1]
        decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
        new_lm_head_decoder = self.add_weight(shape=(new_num_tokens, old_embedding_dim), initializer='zeros', trainable=True, name=old_lm_head_decoder.name.split(':')[0])
        init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())
        new_lm_head_decoder.assign(init_decoder)
    return new_lm_head_decoder