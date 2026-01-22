import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def _get_model_ops_and_info(f_input):
    """Retrieve the root (top level) operators of a model and their corresponding compatibility info.

    These root operators can call other operators within them (traced ops), and
    a root op can call many different traced ops depending on internal code paths in the root op.
    These traced ops are not returned by this function. Those operators are abstracted into the
    runtime as an implementation detail (and the traced ops themselves can also call other operators)
    making retrieving them difficult and their value from this api negligible since they will differ
    between which runtime version the model is run on. Because of this, there is a false positive this
    api can't prevent in a compatibility usecase. All the root ops of a model are present in a
    target runtime, but not all the traced ops are which prevents a model from being able to run.
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        Operators and info: A Dictionary mapping strings (the qualified names of the root operators)
        of the model to their OperatorInfo structs.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_model_ops_and_info

        # Get bytecode version from a saved file path
        ops_and_info = _get_model_ops_and_info("path/to/model.ptl")

    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f'The provided filename {f_input} does not exist')
        if os.path.isdir(f_input):
            raise ValueError(f'The provided filename {f_input} is a directory')
    if isinstance(f_input, (str, pathlib.Path)):
        return torch._C._get_model_ops_and_info(str(f_input))
    else:
        return torch._C._get_model_ops_and_info(f_input.read())