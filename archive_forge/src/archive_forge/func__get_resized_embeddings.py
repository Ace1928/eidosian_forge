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
def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
    """
        Build a resized Embedding weights from a provided token Embedding weights. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`tf.Variable`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `tf.Variable` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is
            `None`
        """
    old_embedding_dim = shape_list(old_embeddings)[1]
    init_range = getattr(self.config, 'initializer_range', 0.02)
    embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
    new_embeddings = self.add_weight(name=old_embeddings.name.split(':')[0], shape=[new_num_tokens, old_embedding_dim], initializer=get_initializer(init_range), dtype=tf.float32)
    init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())
    new_embeddings.assign(init_embeddings)
    return new_embeddings