import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummySpeechT5InputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ('output_sequence', 'speaker_embeddings', 'spectrogram')

    def __init__(self, task: str, normalized_config: NormalizedConfig, sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], **kwargs):
        self.task = task
        self.batch_size = 1
        self.sequence_length = sequence_length
        self.speaker_embedding_dim = normalized_config.speaker_embedding_dim
        self.num_mel_bins = normalized_config.num_mel_bins

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name == 'output_sequence':
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        elif input_name == 'speaker_embeddings':
            shape = [self.batch_size, self.speaker_embedding_dim]
        elif input_name == 'spectrogram':
            shape = [20, self.num_mel_bins]
        else:
            raise ValueError(f'Unsupported input {input_name} for DummySpeechT5InputGenerator')
        return self.random_float_tensor(shape=shape, min_value=0, max_value=1, framework=framework, dtype=float_dtype)