from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class EntryTypesTest(ParserTest, TestCase):
    input_string = u'\n        Testing what are allowed for entry types\n\n        These are OK\n        @somename{an_id,}\n        @t2{another_id,}\n        @t@{again_id,}\n        @t+{aa1_id,}\n        @_t{aa2_id,}\n\n        These ones not\n        @2thou{further_id,}\n        @some name{id3,}\n        @some#{id4,}\n        @some%{id4,}\n    '
    correct_result = BibliographyData([('an_id', Entry('somename')), ('another_id', Entry('t2')), ('again_id', Entry('t@')), ('aa1_id', Entry('t+')), ('aa2_id', Entry('_t'))])
    errors = ['syntax error in line 12: a valid name expected', "syntax error in line 13: '(' or '{' expected", "syntax error in line 14: '(' or '{' expected", "syntax error in line 15: '(' or '{' expected"]