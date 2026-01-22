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
class PickleIO(DatabaseIO):

    def __init__(self, protocol):
        super(PickleIO, self).__init__()
        self.protocol = protocol

    def __repr__(self):
        return '{}(protocol={!r})'.format(type(self).__name__, self.protocol)

    def serialize(self, bib_data):
        return pickle.dumps(bib_data, protocol=self.protocol)

    def deserialize(self, pickled_data):
        return pickle.loads(pickled_data)