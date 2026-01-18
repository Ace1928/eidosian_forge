import dill
from functools import partial
import warnings
def test_partials():
    assert copy(SubMachine(), byref=True)
    assert copy(SubMachine(), byref=True, recurse=True)
    assert copy(SubMachine(), recurse=True)
    assert copy(SubMachine())