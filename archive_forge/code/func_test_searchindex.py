from __future__ import absolute_import, print_function, division
import os
import tempfile
import pytest
from petl.test.helpers import ieq
import petl as etl
from petl.io.whoosh import fromtextindex, totextindex, appendtextindex, \
def test_searchindex():
    dirname = tempfile.mkdtemp()
    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
    ix = create_in(dirname, schema)
    writer = ix.writer()
    writer.add_document(title=u'Oranges', path=u'/a', content=u"This is the first document we've added!")
    writer.add_document(title=u'Apples', path=u'/b', content=u'The second document is even more interesting!')
    writer.commit()
    expect = ((u'path', u'title'), (u'/a', u'Oranges'))
    actual = searchtextindex(dirname, 'oranges')
    ieq(expect, actual)
    actual = searchtextindex(dirname, 'add*')
    ieq(expect, actual)
    expect = ((u'path', u'title'), (u'/a', u'Oranges'), (u'/b', u'Apples'))
    actual = searchtextindex(dirname, 'doc*')
    ieq(expect, actual)