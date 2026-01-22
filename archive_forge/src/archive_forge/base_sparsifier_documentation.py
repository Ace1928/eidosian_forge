import abc
import copy
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, List, Type
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
Converts submodules in input module to a different module according to `mapping`
        by calling `from_dense` method on the target module class
        Args:
            module: input module
            mapping: a dictionary that maps from source module type to target
                module type, can be overwritten to allow swapping user defined
                Modules
            inplace: carry out model transformations in-place, the original module
                is mutated
        