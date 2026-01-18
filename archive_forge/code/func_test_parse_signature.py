import pytest
from jeepney.low_level import *
def test_parse_signature():
    sig = parse_signature(list('(a{sv}(oayays)b)'))
    print(sig)
    assert sig == Struct([Array(DictEntry([simple_types['s'], Variant()])), Struct([simple_types['o'], Array(simple_types['y']), Array(simple_types['y']), simple_types['s']]), simple_types['b']])