import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@dataclass
class _RunDefaults:
    operators_to_quantize: Optional[List[str]] = field(default_factory=lambda: ['Add', 'MatMul'], metadata={'description': 'Operators to quantize, doing no modifications to others (default: `["Add", "MatMul"]`).'})
    node_exclusion: Optional[List[str]] = field(default_factory=lambda: ['layernorm', 'gelu', 'residual', 'gather', 'softmax'], metadata={'description': "Specific nodes to exclude from being quantized (default: `['layernorm', 'gelu', 'residual', 'gather', 'softmax']`)."})
    per_channel: Optional[bool] = field(default=False, metadata={'description': 'Whether to quantize per channel (default: `False`).'})
    calibration: Optional[Calibration] = field(default=None, metadata={'description': 'Calibration parameters, in case static quantization is used.'})
    task_args: Optional[TaskArgs] = field(default=None, metadata={'description': 'Task-specific arguments (default: `None`).'})
    aware_training: Optional[bool] = field(default=False, metadata={'description': 'Whether the quantization is to be done with Quantization-Aware Training (not supported).'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'description': 'Maximum number of samples to use from the evaluation dataset for evaluation.'})
    time_benchmark_args: Optional[BenchmarkTimeArgs] = field(default=BenchmarkTimeArgs(), metadata={'description': 'Parameters related to time benchmark.'})