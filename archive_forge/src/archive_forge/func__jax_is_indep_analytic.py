import warnings
import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode
from pennylane import numpy as pnp
def _jax_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using JAX.

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

    In JAX, we test this by constructing the VJP of the passed function
    and inspecting its signature.
    The first argument of the output of ``jax.vjp`` is a ``Partial``.
    If *any* processing happens to any input, the arguments of that
    ``Partial`` are unequal to ``((),)`.
    Functions that depend on the input in a trivial manner, i.e., without
    processing it, will go undetected by this. Therefore we also
    test the arguments of the *function* of the above ``Partial``.
    The first of these arguments is a list of tuples and if the
    first entry of the first tuple is not ``None``, the input arguments
    are detected to actually feed into the output.

    .. warning::

        This is an experimental function and unknown edge
        cases may exist to this two-stage test.
    """
    import jax
    mapped_func = lambda *_args: func(*_args, **kwargs)
    _vjp = jax.vjp(mapped_func, *args)[1]
    if _vjp.args[0].args != ((),):
        return False
    if _vjp.args[0].func.args[0][0][0] is not None:
        return False
    return True