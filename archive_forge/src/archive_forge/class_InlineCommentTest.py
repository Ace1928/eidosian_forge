from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class InlineCommentTest(ParserTest, TestCase):
    input_string = u'\n        "some text" causes an error like this\n        ``You\'re missing a field name---line 6 of file bibs/inline_comment.bib``\n        for all 3 of the % some text occurences below; in each case the parser keeps\n        what it has up till that point and skips, so that it correctly gets the last\n        entry.\n        @article{Me2010,}\n        @article{Me2011,\n            author="Brett-like, Matthew",\n        % some text\n            title="Another article"}\n        @article{Me2012, % some text\n            author="Real Brett"}\n        This one correctly read\n        @article{Me2013,}\n    '
    correct_result = BibliographyData([('Me2010', Entry('article')), ('Me2011', Entry('article', persons={'author': [Person(first='Matthew', last='Brett-like')]})), ('Me2012', Entry('article')), ('Me2013', Entry('article'))])
    errors = ["syntax error in line 10: '}' expected", "syntax error in line 12: '}' expected"]