import warnings
import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode
from pennylane import numpy as pnp
def _autograd_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using Autograd.

    Args:
        func (callable): Function to test for independence
        args (tuple): Arguments for the function with respect to which
            to test for independence
        kwargs (dict): Keyword arguments for the function at which
            (but not with respect to which) to test for independence

    Returns:
        bool: Whether the function seems to not depend on it ``args``
        analytically. That is, an output of ``True`` means that the
        ``args`` do *not* feed into the output.

    In Autograd, we test this by sending a ``Box`` through the function and
    testing whether the output is again a ``Box`` and on the same trace as
    the input ``Box``. This means that we can trace actual *independence*
    of the output from the input, not only whether the passed function is
    constant.
    The code is adapted from
    `autograd.tracer.py::trace
    <https://github.com/HIPS/autograd/blob/master/autograd/tracer.py#L7>`__.
    """
    node = VJPNode.new_root()
    with trace_stack.new_trace() as t:
        start_box = new_box(args, t, node)
        end_box = func(*start_box, **kwargs)
    if type(end_box) in [tuple, list]:
        if any((isbox(_end) and _end._trace == start_box._trace for _end in end_box)):
            return False
    elif isinstance(end_box, np.ndarray):
        if end_box.ndim == 0:
            end_box = [end_box.item()]
        if any((isbox(_end) and _end._trace == start_box._trace for _end in end_box)):
            return False
    elif isbox(end_box) and end_box._trace == start_box._trace:
        return False
    return True