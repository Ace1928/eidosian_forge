import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class ArgentinianPostalCode(Regex):
    """
    Argentinian Postal codes.

    ::

        >>> ArgentinianPostalCode.to_python('C1070AAM')
        'C1070AAM'
        >>> ArgentinianPostalCode.to_python('c 1070 aam')
        'C1070AAM'
        >>> ArgentinianPostalCode.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (LnnnnLLL)
    """
    format = _('LnnnnLLL')
    regex = re.compile('^([a-zA-Z]{1})\\s*(\\d{4})\\s*([a-zA-Z]{3})$')
    strip = True
    messages = dict(invalid=_('Please enter a zip code (%(format)s)'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self.regex.search(value)
        if not match:
            raise Invalid(self.message('invalid', state, format=self.format), value, state)
        return '%s%s%s' % (match.group(1).upper(), match.group(2), match.group(3).upper())