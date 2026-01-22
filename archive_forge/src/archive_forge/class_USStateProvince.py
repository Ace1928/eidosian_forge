import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class USStateProvince(FancyValidator):
    """
    Valid state or province code (two-letter).

    Well, for now I don't know the province codes, but it does state
    codes.  Give your own `states` list to validate other state-like
    codes; give `extra_states` to add values without losing the
    current state values.

    ::

        >>> s = USStateProvince('XX')
        >>> s.to_python('IL')
        'IL'
        >>> s.to_python('XX')
        'XX'
        >>> s.to_python('xx')
        'XX'
        >>> s.to_python('YY')
        Traceback (most recent call last):
            ...
        Invalid: That is not a valid state code
    """
    states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IN', 'IL', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    extra_states = []
    __unpackargs__ = ('extra_states',)
    messages = dict(empty=_('Please enter a state code'), wrongLength=_('Please enter a state code with TWO letters'), invalid=_('That is not a valid state code'))

    def _validate_python(self, value, state):
        value = str(value).strip().upper()
        if not value:
            raise Invalid(self.message('empty', state), value, state)
        if not value or len(value) != 2:
            raise Invalid(self.message('wrongLength', state), value, state)
        if value not in self.states and (not (self.extra_states and value in self.extra_states)):
            raise Invalid(self.message('invalid', state), value, state)

    def _convert_to_python(self, value, state):
        return str(value).strip().upper()