import numbers as abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IReal(IComplex):
    abc = abc.Real

    @optional
    def __complex__():
        """
        Rarely implemented, even in builtin types.
        """
    __floor__ = __ceil__ = __complex__