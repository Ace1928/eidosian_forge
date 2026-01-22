from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class WindowsNewlineTest(ParserTest, TestCase):
    input_strings = [u"'@Article\r\n\r\n\r\n}\r\n'"]
    correct_result = BibliographyData()
    errors = ["syntax error in line 4: '(' or '{' expected"]