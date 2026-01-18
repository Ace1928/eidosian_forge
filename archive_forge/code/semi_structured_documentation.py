import warnings
from collections import namedtuple
from typing import Any, Optional
import torch
Overload __torch_dispatch__ to use torch._sparse_semi_structured_linear.

        `torch.structured_sparse_linear` uses accelerated sparse CUTLASS kernels.
        In the future we plan to also add in support for cuSPARSELt kernels.

        Args:
            func: The function being dispatched.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation.

        Raises:
            NotImplementedError: If the dispatched operation is not implemented.
        