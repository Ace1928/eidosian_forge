import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):

    def __init__(self, task: str, normalized_config: NormalizedTextConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, batch_size=batch_size, sequence_length=sequence_length, random_batch_size_range=random_batch_size_range, random_sequence_length_range=random_sequence_length_range)
        self.num_key_value_heads = normalized_config.num_key_value_heads

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        shape = (self.batch_size, self.num_key_value_heads, self.sequence_length, self.hidden_size // self.num_attention_heads)
        return [(self.random_float_tensor(shape, framework=framework, dtype=float_dtype), self.random_float_tensor(shape, framework=framework, dtype=float_dtype)) for _ in range(self.num_layers)]