from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods.

    Examples
    --------
    >>> class Foo:
    ...     @classmethod
    ...     def classmethod(cls):
    ...         raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
    ...     def method(self):
    ...         raise pd.errors.AbstractMethodError(self)
    >>> test = Foo.classmethod()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo

    >>> test2 = Foo().method()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo
    """

    def __init__(self, class_instance, methodtype: str='method') -> None:
        types = {'method', 'classmethod', 'staticmethod', 'property'}
        if methodtype not in types:
            raise ValueError(f'methodtype must be one of {methodtype}, got {types} instead.')
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        if self.methodtype == 'classmethod':
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f'This {self.methodtype} must be defined in the concrete class {name}'