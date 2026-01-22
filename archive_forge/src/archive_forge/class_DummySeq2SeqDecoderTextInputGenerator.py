import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummySeq2SeqDecoderTextInputGenerator(DummyDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ('decoder_input_ids', 'decoder_attention_mask', 'encoder_outputs', 'encoder_hidden_states')

    def __init__(self, task: str, normalized_config: NormalizedTextConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], num_choices: int=DEFAULT_DUMMY_SHAPES['num_choices'], random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, random_num_choices_range: Optional[Tuple[int, int]]=None, **kwargs):
        super().__init__(task, normalized_config, batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices, random_batch_size_range=random_batch_size_range, random_sequence_length_range=random_sequence_length_range, random_num_choices_range=random_num_choices_range)
        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            self.hidden_size = normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS.hidden_size
        else:
            self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name in ['encoder_outputs', 'encoder_hidden_states']:
            return (self.random_float_tensor(shape=[self.batch_size, self.sequence_length, self.hidden_size], min_value=0, max_value=1, framework=framework, dtype=float_dtype), None, None)
        return super().generate(input_name, framework=framework, int_dtype=int_dtype)