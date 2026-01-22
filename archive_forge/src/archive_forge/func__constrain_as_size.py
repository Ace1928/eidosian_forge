import math
import os
import sys
import platform
import textwrap
import ctypes
import inspect
from ._utils import _import_dotted_name, classproperty
from ._utils import _functionalize_sync as _sync
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union, List
import builtins
from math import e , nan , inf , pi
from ._tensor import Tensor
from .storage import _StorageBase, TypedStorage, _LegacyStorage, UntypedStorage, _warn_typed_storage_removal
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions
from torch.amp import autocast
from ._compile import _disable_dynamo
from .functional import *  # noqa: F403
from torch import cuda as cuda
from torch import cpu as cpu
from torch import mps as mps
from torch import autograd as autograd
from torch.autograd import (
from torch import fft as fft
from torch import futures as futures
from torch import _awaits as _awaits
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
from torch import backends as backends
import torch.utils.data
from torch import __config__ as __config__
from torch import __future__ as __future__
from torch import profiler as profiler
from torch import ao as ao
import torch.nn.quantizable
import torch.nn.quantized
import torch.nn.qat
import torch.nn.intrinsic
from . import _torch_docs, _tensor_docs, _storage_docs
from torch._ops import ops
from torch._classes import classes
import torch._library
from torch import quantization as quantization
from torch import quasirandom as quasirandom
from torch.multiprocessing._atfork import register_after_fork
from ._lobpcg import lobpcg as lobpcg
from torch.utils.dlpack import from_dlpack, to_dlpack
from . import masked
from ._linalg_utils import (  # type: ignore[misc]
from ._linalg_utils import _symeig as symeig  # type: ignore[misc]
from torch import export as export
from torch._higher_order_ops import cond
from . import return_types
from . import library
import torch.fx.experimental.sym_node
from torch import func as func
from torch.func import vmap
from . import _logging
def _constrain_as_size(symbol, min: Optional[builtins.int]=None, max: Optional[builtins.int]=None):
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics which distinguish it from constrain_as_value.
    Specifically, at compile-time, we will unsoundly assume that the resulting int is always >= 2.
    As a result, max value you pass in should always be greater than 2.
    This makes it easier to use the unbacked int in size contexts, as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if broadcasting can occur),
    which will not work on unbacked SymInts. Assuming that the int is >= 2 allows us to
    report False to these tests. Although this is technically unsound,
    in practice we observe that if your program works for all sizes >= 2,
    it probably works for zero and one too. The reason specifically assume size is >= 2 is because
    lot of PyTorch code is specialized for 0 and 1 which could result in not general graphs.
    At runtime, we only assert that the user provided min/max values are respected.

    To demonstrate in a scenario, suppose you do
    ```
    # Case 1
    # This will assume symbol is between [2, inf) at compile time, but [0, inf) at runtime
    constrain_as_size(symbol, min=0)

    # Case 2
    # This will assume symbol is between [2, N] at compile time, but [0, N] at runtime
    constrain_as_size(symbol, min=0, max=N)

    # Case 3
    # This is not valid case as max is <= 2
    constrain_as_size(symbol, min=0, max=1)

    # Case 4
    # This will assume symbol is between [2, inf) at compile time, AND [2, inf) at runtime
    constrain_as_size(symbol, min=2)

    # Case 5
    # This will assume symbol is between [2, inf) at compile time, but [1, inf) at runtime
    constrain_as_size(symbol, min=1)
    ```
    """
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)