import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
def _train_memory(self, model_name: str, batch_size: int, sequence_length: int) -> [Memory, Optional[MemorySummary]]:
    if self.args.is_gpu:
        tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
    strategy = self.args.strategy
    if strategy is None:
        raise ValueError('A device strategy has to be initialized before using TensorFlow.')
    _train = self._prepare_train_func(model_name, batch_size, sequence_length)
    return self._measure_memory(_train)