from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class EmptyDataTest(ParserTest, TestCase):
    input_string = u''
    correct_result = BibliographyData()