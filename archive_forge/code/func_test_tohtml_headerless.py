from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import io
from petl.test.helpers import eq_
from petl.io.html import tohtml
def test_tohtml_headerless():
    table = []
    f = NamedTemporaryFile(delete=False)
    tohtml(table, f.name, encoding='ascii', lineterminator='\n')
    with io.open(f.name, mode='rt', encoding='ascii', newline='') as o:
        actual = o.read()
        expect = u"<table class='petl'>\n<tbody>\n</tbody>\n</table>\n"
        eq_(expect, actual)