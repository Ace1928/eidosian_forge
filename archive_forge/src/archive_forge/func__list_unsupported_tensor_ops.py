from textwrap import dedent
from typing import Any, Dict
import torch.jit
def _list_unsupported_tensor_ops():
    header = '\n\n\nUnsupported Tensor Methods\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n    '
    methods, properties = _gen_unsupported_methods_properties()
    return header + '\n' + methods + '\n\nUnsupported Tensor Properties\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n    ' + '\n' + properties