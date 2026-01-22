import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from .pygit import PyGit
from .sha1_store import SHA1_Store
def _recursive_apply_to_elements(data: Union[List[Any], Dict[str, Any]], fn: Any, names: List[str]) -> None:
    """Helper function to traverse a dict recursively and apply a function to leafs.


    Args:
        data (dict or list):
            A dict or a list and it should only contain dict and list.
        fn (Any):
            A call back function on each element. Signature:
                fn(element: Any, names: List[str]) -> Any
        names (list):
            Stack of names for making the element path.
    """
    if isinstance(data, list):
        for i, _ in enumerate(data):
            names.append(str(i))
            if isinstance(data[i], (list, dict)):
                _recursive_apply_to_elements(data[i], fn, names)
            else:
                data[i] = fn(data[i], names)
            names.pop()
    elif isinstance(data, dict):
        for key in data.keys():
            names.append(str(key))
            if isinstance(data[key], (list, dict)):
                _recursive_apply_to_elements(data[key], fn, names)
            else:
                data[key] = fn(data[key], names)
            names.pop()
    else:
        assert False, f'Unexpected data type: {type(data)}'