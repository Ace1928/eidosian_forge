import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def threaded(func):
    """Apply ``func`` to sub--elements of an object, including :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator allows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`xthreaded` decorator.

    Functions using this decorator must have the following signature::

      @threaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, True)