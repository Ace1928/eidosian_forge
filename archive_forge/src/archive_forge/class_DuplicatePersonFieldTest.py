from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class DuplicatePersonFieldTest(ParserTest, TestCase):
    input_string = u'\n        @article{Me2009,author={Nom de Plume, My}, title="A short story", AUTHoR = {Foo}}\n    '
    correct_result = BibliographyData(entries=[(u'Me2009', Entry(u'article', fields=[(u'title', u'A short story')], persons={u'author': [Person(u'Nom de Plume, My')]}))])
    errors = ['entry with key Me2009 has a duplicate AUTHoR field']