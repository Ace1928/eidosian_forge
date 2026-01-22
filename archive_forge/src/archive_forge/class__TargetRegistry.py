from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
class _TargetRegistry(DelayedRegistry):

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            msg = "No target is registered against '{}', known targets:\n{}"
            known = '\n'.join([f'{k: <{10}} -> {v}' for k, v in target_registry.items()])
            raise NumbaValueError(msg.format(item, known)) from None