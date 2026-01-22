import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktToken:
    """Stores a token of text with annotations produced during
    sentence boundary detection."""
    _properties = ['parastart', 'linestart', 'sentbreak', 'abbr', 'ellipsis']
    __slots__ = ['tok', 'type', 'period_final'] + _properties

    def __init__(self, tok, **params):
        self.tok = tok
        self.type = self._get_type(tok)
        self.period_final = tok.endswith('.')
        for prop in self._properties:
            setattr(self, prop, None)
        for k in params:
            setattr(self, k, params[k])
    _RE_ELLIPSIS = re.compile('\\.\\.+$')
    _RE_NUMERIC = re.compile('^-?[\\.,]?\\d[\\d,\\.-]*\\.?$')
    _RE_INITIAL = re.compile('[^\\W\\d]\\.$', re.UNICODE)
    _RE_ALPHA = re.compile('[^\\W\\d]+$', re.UNICODE)

    def _get_type(self, tok):
        """Returns a case-normalized representation of the token."""
        return self._RE_NUMERIC.sub('##number##', tok.lower())

    @property
    def type_no_period(self):
        """
        The type with its final period removed if it has one.
        """
        if len(self.type) > 1 and self.type[-1] == '.':
            return self.type[:-1]
        return self.type

    @property
    def type_no_sentperiod(self):
        """
        The type with its final period removed if it is marked as a
        sentence break.
        """
        if self.sentbreak:
            return self.type_no_period
        return self.type

    @property
    def first_upper(self):
        """True if the token's first character is uppercase."""
        return self.tok[0].isupper()

    @property
    def first_lower(self):
        """True if the token's first character is lowercase."""
        return self.tok[0].islower()

    @property
    def first_case(self):
        if self.first_lower:
            return 'lower'
        if self.first_upper:
            return 'upper'
        return 'none'

    @property
    def is_ellipsis(self):
        """True if the token text is that of an ellipsis."""
        return self._RE_ELLIPSIS.match(self.tok)

    @property
    def is_number(self):
        """True if the token text is that of a number."""
        return self.type.startswith('##number##')

    @property
    def is_initial(self):
        """True if the token text is that of an initial."""
        return self._RE_INITIAL.match(self.tok)

    @property
    def is_alpha(self):
        """True if the token text is all alphabetic."""
        return self._RE_ALPHA.match(self.tok)

    @property
    def is_non_punct(self):
        """True if the token is either a number or is alphabetic."""
        return _re_non_punct.search(self.type)

    def __repr__(self):
        """
        A string representation of the token that can reproduce it
        with eval(), which lists all the token's non-default
        annotations.
        """
        typestr = ' type=%s,' % repr(self.type) if self.type != self.tok else ''
        propvals = ', '.join((f'{p}={repr(getattr(self, p))}' for p in self._properties if getattr(self, p)))
        return '{}({},{} {})'.format(self.__class__.__name__, repr(self.tok), typestr, propvals)

    def __str__(self):
        """
        A string representation akin to that used by Kiss and Strunk.
        """
        res = self.tok
        if self.abbr:
            res += '<A>'
        if self.ellipsis:
            res += '<E>'
        if self.sentbreak:
            res += '<S>'
        return res