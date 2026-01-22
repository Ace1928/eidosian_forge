import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
@dataclasses.dataclass
class AllowedObjects:
    """
    Track the objects, object id - name pairs, and name - dynamo wrapping rule pairs
    from the heuristic defined in `gen_allowed_objs_and_ids`.
    TODO: Remove the overalp/duplication between these fields
    after allowed_functions refactor is done.
    """
    object_ids: Dict[int, str]
    ctx_mamager_classes: Set[Any]
    c_binding_in_graph_functions: Set[Any]
    non_c_binding_in_graph_functions: Set[Any]
    name_rule_map: Dict[str, Any]