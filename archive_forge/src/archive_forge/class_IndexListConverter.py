import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class IndexListConverter(FancyValidator):
    """
    Converts a index (which may be a string like '2') to the value in
    the given list.

    Examples::

        >>> index = IndexListConverter(['zero', 'one', 'two'])
        >>> index.to_python(0)
        'zero'
        >>> index.from_python('zero')
        0
        >>> index.to_python('1')
        'one'
        >>> index.to_python(5)
        Traceback (most recent call last):
        Invalid: Index out of range
        >>> index(not_empty=True).to_python(None)
        Traceback (most recent call last):
        Invalid: Please enter a value
        >>> index.from_python('five')
        Traceback (most recent call last):
        Invalid: Item 'five' was not found in the list
    """
    list = None
    __unpackargs__ = ('list',)
    messages = dict(integer=_('Must be an integer index'), outOfRange=_('Index out of range'), notFound=_('Item %(value)s was not found in the list'))

    def _convert_to_python(self, value, state):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise Invalid(self.message('integer', state), value, state)
        try:
            return self.list[value]
        except IndexError:
            raise Invalid(self.message('outOfRange', state), value, state)

    def _convert_from_python(self, value, state):
        for i, v in enumerate(self.list):
            if v == value:
                return i
        raise Invalid(self.message('notFound', state, value=repr(value)), value, state)