from .._utils import set_module
@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """

    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)

    def __str__(self):
        return 'ufunc {!r} did not contain a loop with signature matching types {!r} -> {!r}'.format(self.ufunc.__name__, _unpack_tuple(self.dtypes[:self.ufunc.nin]), _unpack_tuple(self.dtypes[self.ufunc.nin:]))