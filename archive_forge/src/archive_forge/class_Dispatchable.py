import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
class Dispatchable:
    """
    A utility class which marks an argument with a specific dispatch type.


    Attributes
    ----------
    value
        The value of the Dispatchable.

    type
        The type of the Dispatchable.

    Examples
    --------
    >>> x = Dispatchable(1, str)
    >>> x
    <Dispatchable: type=<class 'str'>, value=1>

    See Also
    --------
    all_of_type
        Marks all unmarked parameters of a function.

    mark_as
        Allows one to create a utility function to mark as a given type.
    """

    def __init__(self, value, dispatch_type, coercible=True):
        self.value = value
        self.type = dispatch_type
        self.coercible = coercible

    def __getitem__(self, index):
        return (self.type, self.value)[index]

    def __str__(self):
        return f'<{type(self).__name__}: type={self.type!r}, value={self.value!r}>'
    __repr__ = __str__