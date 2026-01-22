import contextlib
import pennylane as qml
from pennylane.operation import (
@contextlib.contextmanager
def _custom_decomp_context(custom_decomps):
    """A context manager for applying custom decompositions of operations."""

    @contextlib.contextmanager
    def _custom_decomposition(obj, fn):
        if isinstance(obj, str):
            obj = getattr(qml, obj)
        original_decomp_method = obj.compute_decomposition
        try:
            obj.compute_decomposition = staticmethod(fn)
            yield
        finally:
            obj.compute_decomposition = staticmethod(original_decomp_method)
    try:
        with contextlib.ExitStack() as stack:
            for obj, fn in custom_decomps.items():
                stack.enter_context(_custom_decomposition(obj, fn))
            stack = stack.pop_all()
        yield
    finally:
        stack.close()