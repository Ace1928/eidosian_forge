import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def _func_wrapper(loop_vars):
    """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (None or tuple of step_outputs, tuple of new_loop_vars)
        """
    step_output, new_loop_vars = func(*loop_vars)
    if step_output is None:
        step_output = []
    if new_loop_vars is None:
        new_loop_vars = []
    if isinstance(step_output, tuple):
        step_output = list(step_output)
    if isinstance(new_loop_vars, tuple):
        new_loop_vars = list(new_loop_vars)
    new_loop_vars = _as_list(new_loop_vars)
    if len(loop_vars) != len(new_loop_vars):
        raise ValueError('The length of loop_vars should be consistent during the loop')
    return (step_output, new_loop_vars)