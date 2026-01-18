import pickle
import numpy as np
import pytest
import cirq
def test_pickled_hash():
    q = cirq.GridQubit(3, 4)
    q_bad = cirq.GridQubit(3, 4)
    _ = hash(q_bad)
    q_bad._hash = q_bad._hash + 1
    assert q_bad == q
    assert hash(q_bad) != hash(q)
    data = pickle.dumps(q_bad)
    q_ok = pickle.loads(data)
    assert q_ok == q
    assert hash(q_ok) == hash(q)