import collections
from .utils import ExplicitEnum, is_torch_available, logging
class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = 'underflow_overflow'
    TPU_METRICS_DEBUG = 'tpu_metrics_debug'