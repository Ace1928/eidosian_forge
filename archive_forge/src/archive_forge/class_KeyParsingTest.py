from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class KeyParsingTest(ParserTest, TestCase):
    input_string = u'\n        # will not work as expected\n        @article(test(parens1))\n\n        # works fine\n        @article(test(parens2),)\n\n        # works fine\n        @article{test(braces1)}\n\n        # also works\n        @article{test(braces2),}\n    '
    correct_result = BibliographyData([('test(parens1))', Entry('article')), ('test(parens2)', Entry('article')), ('test(braces1)', Entry('article')), ('test(braces2)', Entry('article'))])
    errors = ["syntax error in line 5: ')' expected"]