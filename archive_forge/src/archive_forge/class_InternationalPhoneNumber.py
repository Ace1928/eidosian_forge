import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class InternationalPhoneNumber(FancyValidator):
    """
    Validates, and converts phone numbers to +##-###-#######.
    Adapted from RFC 3966

    @param  default_cc      country code for prepending if none is provided
                            can be a paramerless callable

    ::

        >>> c = InternationalPhoneNumber(default_cc=lambda: 49)
        >>> c.to_python('0555/8114100')
        '+49-555-8114100'
        >>> p = InternationalPhoneNumber(default_cc=49)
        >>> p.to_python('333-3333')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a number, with area code, in the form +##-###-#######.
        >>> p.to_python('0555/4860-300')
        '+49-555-4860-300'
        >>> p.to_python('0555-49924-51')
        '+49-555-49924-51'
        >>> p.to_python('0555 / 8114100')
        '+49-555-8114100'
        >>> p.to_python('0555/8114100')
        '+49-555-8114100'
        >>> p.to_python('0555 8114100')
        '+49-555-8114100'
        >>> p.to_python(' +49 (0)555 350 60 0')
        '+49-555-35060-0'
        >>> p.to_python('+49 555 350600')
        '+49-555-350600'
        >>> p.to_python('0049/ 555/ 871 82 96')
        '+49-555-87182-96'
        >>> p.to_python('0555-2 50-30')
        '+49-555-250-30'
        >>> p.to_python('0555 43-1200')
        '+49-555-43-1200'
        >>> p.to_python('(05 55)4 94 33 47')
        '+49-555-49433-47'
        >>> p.to_python('(00 48-555)2 31 72 41')
        '+48-555-23172-41'
        >>> p.to_python('+973-555431')
        '+973-555431'
        >>> p.to_python('1-393-555-3939')
        '+1-393-555-3939'
        >>> p.to_python('+43 (1) 55528/0')
        '+43-1-55528-0'
        >>> p.to_python('+43 5555 429 62-0')
        '+43-5555-42962-0'
        >>> p.to_python('00 218 55 33 50 317 321')
        '+218-55-3350317-321'
        >>> p.to_python('+218 (0)55-3636639/38')
        '+218-55-3636639-38'
        >>> p.to_python('032 555555 367')
        '+49-32-555555-367'
        >>> p.to_python('(+86) 555 3876693')
        '+86-555-3876693'
    """
    strip = True
    default_cc = None
    _mark_chars_re = re.compile("[_.!~*'/]")
    _preTransformations = [(re.compile('^(\\(?)(?:00\\s*)(.+)$'), '%s+%s'), (re.compile('^\\(\\s*(\\+?\\d+)\\s*(\\d+)\\s*\\)(.+)$'), '(%s%s)%s'), (re.compile('^\\((\\+?[-\\d]+)\\)\\s?(\\d.+)$'), '%s-%s'), (re.compile('^(?:1-)(\\d+.+)$'), '+1-%s'), (re.compile('^(\\+\\d+)\\s+\\(0\\)\\s*(\\d+.+)$'), '%s-%s'), (re.compile('^([0+]\\d+)[-\\s](\\d+)$'), '%s-%s'), (re.compile('^([0+]\\d+)[-\\s](\\d+)[-\\s](\\d+)$'), '%s-%s-%s')]
    _ccIncluder = [(re.compile('^\\(?0([1-9]\\d*)[-)](\\d.*)$'), '+%d-%s-%s')]
    _postTransformations = [(re.compile('^(\\+\\d+)[-\\s]\\(?(\\d+)\\)?[-\\s](\\d+.+)$'), '%s-%s-%s'), (re.compile('^(.+)\\s(\\d+)$'), '%s-%s')]
    _phoneIsSane = re.compile('^(\\+[1-9]\\d*)-([\\d\\-]+)$')
    messages = dict(phoneFormat=_('Please enter a number, with area code, in the form +##-###-#######.'))

    def _perform_rex_transformation(self, value, transformations):
        for rex, trf in transformations:
            match = rex.search(value)
            if match:
                value = trf % match.groups()
        return value

    def _prepend_country_code(self, value, transformations, country_code):
        for rex, trf in transformations:
            match = rex.search(value)
            if match:
                return trf % ((country_code,) + match.groups())
        return value

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        try:
            value = value.encode('ascii', 'strict')
        except UnicodeEncodeError:
            raise Invalid(self.message('phoneFormat', state), value, state)
        value = value.decode('ascii')
        value = self._mark_chars_re.sub('-', value)
        for f, t in [('  ', ' '), ('--', '-'), (' - ', '-'), ('- ', '-'), (' -', '-')]:
            value = value.replace(f, t)
        value = self._perform_rex_transformation(value, self._preTransformations)
        if self.default_cc:
            if callable(self.default_cc):
                cc = self.default_cc()
            else:
                cc = self.default_cc
            value = self._prepend_country_code(value, self._ccIncluder, cc)
        value = self._perform_rex_transformation(value, self._postTransformations)
        value = value.replace(' ', '')
        if not self._phoneIsSane.search(value):
            raise Invalid(self.message('phoneFormat', state), value, state)
        return value