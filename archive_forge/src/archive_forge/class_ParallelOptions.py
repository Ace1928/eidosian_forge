from abc import ABCMeta, abstractmethod
class ParallelOptions(AbstractOptionValue):
    """
    Options for controlling auto parallelization.
    """
    __slots__ = ('enabled', 'comprehension', 'reduction', 'inplace_binop', 'setitem', 'numpy', 'stencil', 'fusion', 'prange')

    def __init__(self, value):
        if isinstance(value, bool):
            self.enabled = value
            self.comprehension = value
            self.reduction = value
            self.inplace_binop = value
            self.setitem = value
            self.numpy = value
            self.stencil = value
            self.fusion = value
            self.prange = value
        elif isinstance(value, dict):
            self.enabled = True
            self.comprehension = value.pop('comprehension', True)
            self.reduction = value.pop('reduction', True)
            self.inplace_binop = value.pop('inplace_binop', True)
            self.setitem = value.pop('setitem', True)
            self.numpy = value.pop('numpy', True)
            self.stencil = value.pop('stencil', True)
            self.fusion = value.pop('fusion', True)
            self.prange = value.pop('prange', True)
            if value:
                msg = 'Unrecognized parallel options: %s' % value.keys()
                raise NameError(msg)
        elif isinstance(value, ParallelOptions):
            self.enabled = value.enabled
            self.comprehension = value.comprehension
            self.reduction = value.reduction
            self.inplace_binop = value.inplace_binop
            self.setitem = value.setitem
            self.numpy = value.numpy
            self.stencil = value.stencil
            self.fusion = value.fusion
            self.prange = value.prange
        else:
            msg = 'Expect parallel option to be either a bool or a dict'
            raise ValueError(msg)

    def _get_values(self):
        """Get values as dictionary.
        """
        return {k: getattr(self, k) for k in self.__slots__}

    def __eq__(self, other):
        if type(other) is type(self):
            return self._get_values() == other._get_values()
        return NotImplemented

    def encode(self) -> str:
        return ', '.join((f'{k}={v}' for k, v in self._get_values().items()))