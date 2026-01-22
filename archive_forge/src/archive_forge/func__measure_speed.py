import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
def _measure_speed(self, func) -> float:
    with self.args.strategy.scope():
        try:
            if self.args.is_tpu or self.args.use_xla:
                logger.info('Do inference on TPU. Running model 5 times to stabilize compilation')
                timeit.repeat(func, repeat=1, number=5)
            runtimes = timeit.repeat(func, repeat=self.args.repeat, number=10)
            return min(runtimes) / 10.0
        except ResourceExhaustedError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")