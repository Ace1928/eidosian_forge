from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
class UndefinedFunctionType(FunctionType):
    _counter = 0

    def __init__(self, nargs, dispatchers):
        from numba.core.typing.templates import Signature
        signature = Signature(types.undefined, (types.undefined,) * nargs, recvr=None)
        super(UndefinedFunctionType, self).__init__(signature)
        self.dispatchers = dispatchers
        type(self)._counter += 1
        self._key += str(type(self)._counter)

    def get_precise(self):
        """
        Return precise function type if possible.
        """
        for dispatcher in self.dispatchers:
            for cres in dispatcher.overloads.values():
                sig = types.unliteral(cres.signature)
                return FunctionType(sig)
        return self