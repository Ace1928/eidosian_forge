import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class StripField(FancyValidator):
    """
    Take a field from a dictionary, removing the key from the dictionary.

    ``name`` is the key.  The field value and a new copy of the dictionary
    with that field removed are returned.

    >>> StripField('test').to_python({'a': 1, 'test': 2})
    (2, {'a': 1})
    >>> StripField('test').to_python({})
    Traceback (most recent call last):
        ...
    Invalid: The name 'test' is missing

    """
    __unpackargs__ = ('name',)
    messages = dict(missing=_('The name %(name)s is missing'))

    def _convert_to_python(self, valueDict, state):
        v = valueDict.copy()
        try:
            field = v.pop(self.name)
        except KeyError:
            raise Invalid(self.message('missing', state, name=repr(self.name)), valueDict, state)
        return (field, v)

    def is_empty(self, value):
        return False