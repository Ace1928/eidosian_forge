import os
import math
import dill as pickle
def test_pickled_cadder():
    pcadder = pickle.dumps(cadder)
    pcadd5 = pickle.loads(pcadder)(x)
    assert pcadd5(y) == x + y