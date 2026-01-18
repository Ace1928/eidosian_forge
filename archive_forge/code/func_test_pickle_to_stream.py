import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
def test_pickle_to_stream():
    dumpfile = dumpIO(x)
    _x = loadIO(dumpfile)
    assert _x == x