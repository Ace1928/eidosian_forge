import numbers as abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IComplex(INumber):
    abc = abc.Complex

    @optional
    def __complex__():
        """
        Rarely implemented, even in builtin types.
        """