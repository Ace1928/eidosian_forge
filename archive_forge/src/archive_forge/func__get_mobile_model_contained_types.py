import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def _get_mobile_model_contained_types(f_input) -> int:
    """Take a file-like object and return a set of string, like ("int", "Optional").

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        type_list: A set of string, like ("int", "Optional"). These are types used in bytecode.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_mobile_model_contained_types

        # Get type list from a saved file path
        type_list = _get_mobile_model_contained_types("path/to/model.ptl")

    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f'The provided filename {f_input} does not exist')
        if os.path.isdir(f_input):
            raise ValueError(f'The provided filename {f_input} is a directory')
    if isinstance(f_input, (str, pathlib.Path)):
        return torch._C._get_mobile_model_contained_types(str(f_input))
    else:
        return torch._C._get_mobile_model_contained_types_from_buffer(f_input.read())