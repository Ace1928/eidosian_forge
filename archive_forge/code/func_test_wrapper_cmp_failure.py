import itertools
import random
import pytest
import networkx
import cirq
def test_wrapper_cmp_failure():
    with pytest.raises(TypeError):
        _ = object() < cirq.contrib.Unique(1)
    with pytest.raises(TypeError):
        _ = cirq.contrib.Unique(1) < object()