from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.compat import pickle
from petl.test.helpers import ieq
from petl.io.pickle import frompickle, topickle, appendpickle
def test_topickle_headerless():
    table = []
    f = NamedTemporaryFile(delete=False)
    topickle(table, f.name)
    expect = []
    with open(f.name, 'rb') as o:
        ieq(expect, picklereader(o))