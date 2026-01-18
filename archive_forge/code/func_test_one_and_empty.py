import pytest
from preshed.maps import PreshMap
import random
def test_one_and_empty():
    table = PreshMap()
    for i in range(100, 110):
        table[i] = i
        del table[i]
    assert table[0] == None