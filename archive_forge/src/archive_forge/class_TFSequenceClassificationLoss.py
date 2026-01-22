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
class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def hf_compute_loss(self, labels, logits):
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                labels = tf.expand_dims(labels, axis=-1)
        else:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        return loss_fn(labels, logits)