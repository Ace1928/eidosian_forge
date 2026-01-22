from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def abbreviate(self):

    def abbreviate_word(word):
        if word.isalpha():
            return word[0].add_period()
        else:
            return word
    parts = self.split(textutils.delimiter_re)
    return String('').join((abbreviate_word(part) for part in parts))