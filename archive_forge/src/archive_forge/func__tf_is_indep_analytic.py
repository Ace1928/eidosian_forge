import warnings
import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode
from pennylane import numpy as pnp
def _tf_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using TensorFlow.

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

    In TensorFlow, we test this by computing the Jacobian of the output(s)
    with respect to the arguments. If the Jacobian is ``None``, the output(s)
    is/are independent.

    .. note::

        Of all interfaces, this is currently the most robust for the
        ``is_independent`` functionality.
    """
    import tensorflow as tf
    with tf.GradientTape(persistent=True) as tape:
        out = func(*args, **kwargs)
    if isinstance(out, tuple):
        jac = [tape.jacobian(_out, args) for _out in out]
        return all((all((__jac is None for __jac in _jac)) for _jac in jac))
    jac = tape.jacobian(out, args)
    return all((_jac is None for _jac in jac))