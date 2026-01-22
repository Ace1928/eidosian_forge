import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _create_name_sharing_dict(duplicate_weights: DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]], suffix: str='') -> Dict[Tuple[str, int], str]:
    """
    Creates a map mapping old initializer names to new initializer names. As different ONNX models
    may use the same initializer name but need to be mapped to a different new name, the map is actually from
    (old name, model id) to new name.

    Example of initializers with the same name that will need to be mapped to a different one:
    Model 1 with:
    /transformer/Constant_8_output_0 of datatype 1

    Model 2 with:
    /transformer/Constant_8_output_0 of datatype 7

    Args:
        duplicate_weights (`DefaultDict[Tuple[int, bytes]`):

        suffix (`str`, defaults to `""`):
    """
    name_sharing_dict = {}
    used_common_names = {}
    for duplicates in duplicate_weights.values():
        common_name, model_id = duplicates.pop()
        if common_name in used_common_names:
            used_common_names[common_name] += 1
        else:
            used_common_names[common_name] = 0
        duplicates.add((common_name, model_id))
        for k in duplicates:
            assert k not in name_sharing_dict
            name_sharing_dict[k] = f'{common_name}_{suffix}_{used_common_names[common_name]}' if suffix != '' else f'{common_name}'
    return name_sharing_dict