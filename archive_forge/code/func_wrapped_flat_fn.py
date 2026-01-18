import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
@wraps(flat_fn)
def wrapped_flat_fn(*args):
    unpacked_args = _unpack_synthetic_bases(args)
    aliased_args_with_metadata_mutations = [x for i, x in enumerate(unpacked_args) if i in aliased_arg_idx_with_metadata_mutations]
    if len(aliased_args_with_metadata_mutations) > 0:
        return (*flat_fn(*unpacked_args), *aliased_args_with_metadata_mutations)
    else:
        return flat_fn(*unpacked_args)