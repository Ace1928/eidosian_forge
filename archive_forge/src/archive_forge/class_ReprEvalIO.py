from __future__ import absolute_import, unicode_literals
import pickle
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from io import BytesIO, TextIOWrapper
import six
import pytest
from pybtex.database import parse_bytes, parse_string, BibliographyData, Entry
from pybtex.plugin import find_plugin
from .data import reference_data
class ReprEvalIO(DatabaseIO):

    def __repr__(self):
        return '{}()'.format(type(self).__name__)

    def serialize(self, bib_data):
        return repr(bib_data)

    def deserialize(self, repr_value):
        from pybtex.utils import OrderedCaseInsensitiveDict
        from pybtex.database import BibliographyData, Entry, Person
        return eval(repr_value, {'OrderedCaseInsensitiveDict': OrderedCaseInsensitiveDict, 'BibliographyData': BibliographyData, 'Entry': Entry, 'Person': Person})