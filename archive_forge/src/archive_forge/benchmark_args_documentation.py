from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments

        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        