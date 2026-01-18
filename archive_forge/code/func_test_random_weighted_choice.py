import pytest
from networkx.utils import (
def test_random_weighted_choice():
    mapping = {'a': 10, 'b': 0}
    c = weighted_choice(mapping, seed=1)
    c = weighted_choice(mapping)
    assert c == 'a'