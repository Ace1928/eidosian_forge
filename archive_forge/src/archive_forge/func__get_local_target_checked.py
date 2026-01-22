from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
def _get_local_target_checked(tyctx, hwstr, reason):
    """Returns the local target if it is compatible with the given target
    name during a type resolution; otherwise, raises an exception.

    Parameters
    ----------
    tyctx: typing context
    hwstr: str
        target name to check against
    reason: str
        Reason for the resolution. Expects a noun.
    Returns
    -------
    target_hw : Target

    Raises
    ------
    InternalTargetMismatchError
    """
    hw_clazz = resolve_target_str(hwstr)
    target_hw = get_local_target(tyctx)
    if not target_hw.inherits_from(hw_clazz):
        raise InternalTargetMismatchError(reason, target_hw, hw_clazz)
    return target_hw