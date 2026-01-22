import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
class RecordLoadStore(V.KernelFormatterHandler):

    def __init__(self, var_ranges: VarRanges, normalize: bool):
        parent_handler = _RecordLoadStoreInner(var_ranges=var_ranges, normalize=normalize)
        parent_handler = _OpCounter(parent_handler)
        super().__init__(parent_handler=parent_handler)