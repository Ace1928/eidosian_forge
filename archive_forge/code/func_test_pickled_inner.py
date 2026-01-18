import os
import math
import dill as pickle
def test_pickled_inner():
    add5 = adder(x)
    pinner = pickle.dumps(add5)
    p5add = pickle.loads(pinner)
    assert p5add(y) == x + y