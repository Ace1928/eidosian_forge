import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
def _schemas_match(functional_schema, inplace_schema):
    names_match = inplace_schema.name.endswith('_') and inplace_schema.name[:-1] == functional_schema.name
    arg_types_match = len(functional_schema.arguments) == len(inplace_schema.arguments) and all((a1.type == a2.type for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments)))
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    assert all((a.alias_info is None for a in inplace_schema.arguments[1:]))
    return names_match and arg_types_match