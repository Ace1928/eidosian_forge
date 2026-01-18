import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
def test_two_arg_functions():
    for obj in [add]:
        pyfile = dumpIO_source(obj, alias='_obj')
        _obj = loadIO_source(pyfile)
        assert _obj(4, 2) == obj(4, 2)