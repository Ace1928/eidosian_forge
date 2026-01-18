import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_pickling(self):
    f = BytesIO()
    pickle.dump(rlc.OrdDict([('a', 1), ('b', 2)]), f)
    f.seek(0)
    od = pickle.load(f)
    assert od['a'] == 1
    assert od.index('a') == 0
    assert od['b'] == 2
    assert od.index('b') == 1