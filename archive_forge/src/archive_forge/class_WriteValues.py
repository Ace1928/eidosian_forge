import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
@register_operator
class WriteValues(BaseOperator):
    OPERATOR = get_xformers_operator('write_values')
    OPERATOR_CATEGORY = 'sequence_parallel_fused'
    NAME = 'write_values'