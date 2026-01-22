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
def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int]=None) -> keras.layers.Embedding:
    """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
    if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
        return self.get_input_embeddings()
    model_embeds = self._v2_resize_token_embeddings(new_num_tokens)
    self.config.vocab_size = new_num_tokens
    return model_embeds