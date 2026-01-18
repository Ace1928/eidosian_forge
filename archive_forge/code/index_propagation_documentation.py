import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    