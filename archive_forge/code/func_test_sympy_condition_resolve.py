import re
import pytest
import sympy
import cirq
def test_sympy_condition_resolve():

    def resolve(records):
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return init_sympy_condition.resolve(classical_data)
    assert resolve({'0:a': [[1]]})
    assert resolve({'0:a': [[2]]})
    assert resolve({'0:a': [[0, 1]]})
    assert resolve({'0:a': [[1, 0]]})
    assert not resolve({'0:a': [[0]]})
    assert not resolve({'0:a': [[0, 0]]})
    assert not resolve({'0:a': [[]]})
    assert not resolve({'0:a': [[0]], 'b': [[1]]})
    with pytest.raises(ValueError, match=re.escape("Measurement keys ['0:a'] missing when testing classical control")):
        _ = resolve({})
    with pytest.raises(ValueError, match=re.escape("Measurement keys ['0:a'] missing when testing classical control")):
        _ = resolve({'0:b': [[1]]})