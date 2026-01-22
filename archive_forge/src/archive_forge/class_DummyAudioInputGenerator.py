import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyAudioInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ('input_features', 'input_values')

    def __init__(self, task: str, normalized_config: NormalizedConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], feature_size: int=DEFAULT_DUMMY_SHAPES['feature_size'], nb_max_frames: int=DEFAULT_DUMMY_SHAPES['nb_max_frames'], audio_sequence_length: int=DEFAULT_DUMMY_SHAPES['audio_sequence_length'], **kwargs):
        self.task = task
        self.normalized_config = normalized_config
        if hasattr(self.normalized_config, 'feature_size'):
            self.feature_size = self.normalized_config.feature_size
        else:
            self.feature_size = feature_size
        self.nb_max_frames = nb_max_frames
        self.batch_size = batch_size
        self.sequence_length = audio_sequence_length

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name == 'input_values':
            return self.random_float_tensor(shape=[self.batch_size, self.sequence_length], min_value=-1, max_value=1, framework=framework, dtype=float_dtype)
        else:
            return self.random_float_tensor(shape=[self.batch_size, self.feature_size, self.nb_max_frames], min_value=-1, max_value=1, framework=framework, dtype=float_dtype)