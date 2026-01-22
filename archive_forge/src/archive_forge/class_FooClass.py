import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
class FooClass(object):
    """FooClass

    Example:

    >>> 1+1
    2
    """

    @skip_doctest
    def __init__(self, x):
        """Make a FooClass.

        Example:

        >>> f = FooClass(3)
        junk
        """
        print('Making a FooClass.')
        self.x = x

    @skip_doctest
    def bar(self, y):
        """Example:

        >>> ff = FooClass(3)
        >>> ff.bar(0)
        boom!
        >>> 1/0
        bam!
        """
        return 1 / y

    def baz(self, y):
        """Example:

        >>> ff2 = FooClass(3)
        Making a FooClass.
        >>> ff2.baz(3)
        True
        """
        return self.x == y