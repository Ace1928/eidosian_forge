import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def xthreaded(func):
    """Apply ``func`` to sub--elements of an object, excluding :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator disallows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`threaded` decorator.

    Functions using this decorator must have the following signature::

      @xthreaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, False)