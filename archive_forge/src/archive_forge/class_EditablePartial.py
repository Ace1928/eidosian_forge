import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
class EditablePartial(Generic[T]):
    """
    Acts like a functools.partial, but can be edited. In other words, it represents a type that hasn't yet been
    constructed.
    """

    def __init__(self, func: Callable[..., T], args: list, kwargs: dict):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_call(cls, func: Callable[..., T], *args, **kwargs) -> 'EditablePartial[T]':
        """
        If you call this function in the same way that you would call the constructor, it will store the arguments
        as you expect. For example EditablePartial.from_call(Fraction, 1, 3)() == Fraction(1, 3)
        """
        return EditablePartial(func=func, args=list(args), kwargs=kwargs)

    @property
    def name(self):
        return self.kwargs['name']

    def __call__(self) -> T:
        """
        Evaluate the partial and return the result
        """
        args = self.args.copy()
        kwargs = self.kwargs.copy()
        arg_spec = inspect.getfullargspec(self.func)
        if arg_spec.varargs in self.kwargs:
            args += kwargs.pop(arg_spec.varargs)
        return self.func(*args, **kwargs)