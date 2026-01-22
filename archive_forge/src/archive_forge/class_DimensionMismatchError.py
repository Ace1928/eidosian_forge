import dis
import inspect
from typing import Sequence, Union
import torch
import functorch._C
from functorch._C import dim as _C
from .tree_map import tree_flatten, tree_map
from .wrap_type import wrap_type
from . import op_properties
class DimensionMismatchError(Exception):
    pass