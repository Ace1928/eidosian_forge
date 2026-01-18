import pytest
import cirq
@pytest.mark.parametrize('p', [0.0, 0.1, 0.6, 1.0])
def test_validate_probability_valid(p):
    assert p == cirq.validate_probability(p, 'p')