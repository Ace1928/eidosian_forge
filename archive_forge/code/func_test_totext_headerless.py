from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import io
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_totext_headerless():
    table = []
    f = NamedTemporaryFile(delete=False)
    prologue = '-- START\n'
    template = '+ {f1}\n'
    epilogue = '-- END\n'
    totext(table, f.name, encoding='ascii', template=template, prologue=prologue, epilogue=epilogue)
    with io.open(f.name, mode='rt', encoding='ascii', newline='') as o:
        actual = o.read()
        expect = prologue + epilogue
        eq_(expect, actual)