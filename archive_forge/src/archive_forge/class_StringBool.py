import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class StringBool(FancyValidator):
    """
    Converts a string to a boolean.

    Values like 'true' and 'false' are considered True and False,
    respectively; anything in ``true_values`` is true, anything in
    ``false_values`` is false, case-insensitive).  The first item of
    those lists is considered the preferred form.

    ::

        >>> s = StringBool()
        >>> s.to_python('yes'), s.to_python('no')
        (True, False)
        >>> s.to_python(1), s.to_python('N')
        (True, False)
        >>> s.to_python('ye')
        Traceback (most recent call last):
            ...
        Invalid: Value should be 'true' or 'false'
    """
    true_values = ['true', 't', 'yes', 'y', 'on', '1']
    false_values = ['false', 'f', 'no', 'n', 'off', '0']
    messages = dict(string=_('Value should be %(true)r or %(false)r'))

    def _convert_to_python(self, value, state):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in self.true_values:
                return True
            if not value or value in self.false_values:
                return False
            raise Invalid(self.message('string', state, true=self.true_values[0], false=self.false_values[0]), value, state)
        return bool(value)

    def _convert_from_python(self, value, state):
        return (self.true_values if value else self.false_values)[0]