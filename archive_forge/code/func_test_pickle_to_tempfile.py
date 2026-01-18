import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
def test_pickle_to_tempfile():
    if not WINDOWS:
        dumpfile = dump(x)
        _x = load(dumpfile)
        assert _x == x