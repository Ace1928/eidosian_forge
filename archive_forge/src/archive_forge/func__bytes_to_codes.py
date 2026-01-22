import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _bytes_to_codes(b):
    a = memoryview(b).cast('I')
    assert a.itemsize == _sre.CODESIZE
    assert len(a) * a.itemsize == len(b)
    return a.tolist()