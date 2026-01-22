import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
def _unflatten_dict(checkpoint: Dict[str, Any], key_map: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """Converts the flat dictionary with keys 'x.y.z...' to a nested dictionary using the provided key map.

    Args:
        checkpoint: The flat checkpoint dictionary.
        key_map: A dictionary that maps the keys in flattened format 'x.y.z...' to a tuple representing
            the index path into the nested dictonary that this function should construct.

    """
    assert checkpoint.keys() == key_map.keys()
    converted: Dict[str, Any] = {}
    for flat_key in checkpoint:
        key_path = key_map[flat_key]
        _set_nested_dict_value(converted, key_path, checkpoint[flat_key])
    return converted