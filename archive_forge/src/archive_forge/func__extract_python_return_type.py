import torch
import torch.fx
import inspect
from typing import Any, Dict, Optional, Tuple
from torch.fx.node import Argument, Target
from torch._jit_internal import boolean_dispatched
from torch.fx.operator_schemas import _torchscript_type_to_python_type
from torch.fx import Transformer
def _extract_python_return_type(self, target: Target) -> Optional[Any]:
    """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
    assert callable(target)
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return None
    return sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else None