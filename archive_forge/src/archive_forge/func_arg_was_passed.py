import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def arg_was_passed(self, arg_name, args, kwargs, inputs_in_args=False):
    """Returns true if argument is present in `args` or `kwargs`.

        Args:
          arg_name: String name of the argument to find.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.

        Returns:
          True if argument with `arg_name` is present in `args` or `kwargs`.
        """
    if not args and (not kwargs):
        return False
    if arg_name in kwargs:
        return True
    call_fn_args = self._arg_names
    if not inputs_in_args:
        call_fn_args = call_fn_args[1:]
    return arg_name in dict(zip(call_fn_args, args))