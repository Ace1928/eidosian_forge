from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class MacrosTest(ParserTest, TestCase):
    input_string = u'\n        @String{and = { and }}\n        @String{etal = and # { {et al.}}}\n        @Article(\n            unknown,\n            author = nobody,\n        )\n        @Article(\n            gsl,\n            author = "Gough, Brian"#etal,\n        )\n    '
    correct_result = BibliographyData([('unknown', Entry('article')), ('gsl', Entry('article', persons={u'author': [Person(u'Gough, Brian'), Person(u'{et al.}')]}))])
    errors = ['undefined string in line 6: nobody']