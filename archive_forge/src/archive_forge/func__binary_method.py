from numpy.core import umath as um
def _binary_method(ufunc, name):
    """Implement a forward binary method with a ufunc, e.g., __add__."""

    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)
    func.__name__ = '__{}__'.format(name)
    return func