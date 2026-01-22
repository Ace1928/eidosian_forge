from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
class WordGenerator(SequenceGenerator):
    """
    Class which generates passwords by randomly choosing from a string of unique characters.

    Parameters
    ----------
    :param chars:
        custom character string to draw from.

    :param charset:
        predefined charset to draw from.

    :param \\*\\*kwds:
        all other keywords passed to the :class:`SequenceGenerator` parent class.

    Attributes
    ----------
    .. autoattribute:: chars
    .. autoattribute:: charset
    .. autoattribute:: default_charsets
    """
    charset = 'ascii_62'
    chars = None

    def __init__(self, chars=None, charset=None, **kwds):
        if chars:
            if charset:
                raise TypeError('`chars` and `charset` are mutually exclusive')
        else:
            if not charset:
                charset = self.charset
                assert charset
            chars = default_charsets[charset]
        self.charset = charset
        chars = to_unicode(chars, param='chars')
        _ensure_unique(chars, param='chars')
        self.chars = chars
        super(WordGenerator, self).__init__(**kwds)

    @memoized_property
    def symbol_count(self):
        return len(self.chars)

    def __next__(self):
        return getrandstr(self.rng, self.chars, self.length)