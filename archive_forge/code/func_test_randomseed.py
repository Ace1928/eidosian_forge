import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_randomseed():
    """
    Ensure that randomseed provides a non-empty string that changes.
    """
    seed_1 = randomseed()
    time.sleep(1)
    seed_2 = randomseed()
    assert isinstance(seed_1, str)
    assert seed_1 != ''
    assert seed_1 != seed_2