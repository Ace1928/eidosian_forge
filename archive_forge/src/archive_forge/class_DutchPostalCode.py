import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class DutchPostalCode(Regex):
    """
    Dutch Postal codes.

    ::

        >>> DutchPostalCode.to_python('1011 PN')
        '1011PN'
        >>> DutchPostalCode.to_python('3400ac')
        '3400AC'
        >>> DutchPostalCode.to_python('12345')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (nnnnLL)
    """
    format = _('nnnnLL')
    regex = re.compile('^(\\d{4})\\s?([a-zA-Z]{2})$')
    strip = True
    messages = dict(invalid=_('Please enter a zip code (%(format)s)'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self.regex.search(value)
        if not match:
            raise Invalid(self.message('invalid', state, format=self.format), value, state)
        return '%s%s' % (match.group(1), match.group(2).upper())