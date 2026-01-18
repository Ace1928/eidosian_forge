import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def vectorize_call(self, broadcast_shape, args, kwargs):
    """Run the function in a for loop.

        A possible extension would be to parallelize the calls.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """
    outputs = []
    for index in np.ndindex(*broadcast_shape):
        current_args = tuple((arg[index] for arg in args))
        current_kwargs = {key: value[index] for key, value in kwargs.items()}
        outputs.append(self.func(*current_args, **current_kwargs))
    return outputs