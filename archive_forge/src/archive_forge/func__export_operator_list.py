import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def _export_operator_list(module: LiteScriptModule):
    """Return a set of root operator names (with overload name) that are used by any method in this mobile module."""
    return torch._C._export_operator_list(module._c)