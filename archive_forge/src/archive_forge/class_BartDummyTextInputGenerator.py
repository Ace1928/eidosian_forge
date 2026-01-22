import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class BartDummyTextInputGenerator(DummyTextInputGenerator):

    def __init__(self, task: str, normalized_config: NormalizedSeq2SeqConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], num_choices: int=DEFAULT_DUMMY_SHAPES['num_choices'], random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, random_num_choices_range: Optional[Tuple[int, int]]=None, force_eos_token_id_presence: bool=True, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices, random_batch_size_range=random_batch_size_range, random_sequence_length_range=random_sequence_length_range, random_num_choices_range=random_num_choices_range)
        self.force_eos_token_id_presence = force_eos_token_id_presence
        self.eos_token_id = normalized_config.eos_token_id

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        int_tensor = super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
        if self.force_eos_token_id_presence and 'input_ids' in input_name and (self.task == 'text-classification'):
            for idx in range(self.batch_size):
                if self.eos_token_id in int_tensor[idx]:
                    continue
                random_idx = random.randint(1, self.sequence_length - 1)
                int_tensor[idx][random_idx] = self.eos_token_id
        return int_tensor