from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat

        Return True if the string ends with the specified suffix,
        otherwise return False.

        suffix can also be a tuple of suffixes to look for.
        return self.value.endswith(text)
        