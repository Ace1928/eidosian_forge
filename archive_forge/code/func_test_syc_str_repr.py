import cirq
import cirq_google
def test_syc_str_repr():
    assert str(cirq_google.PhysicalZTag()) == 'PhysicalZTag()'
    assert repr(cirq_google.PhysicalZTag()) == 'cirq_google.PhysicalZTag()'
    cirq.testing.assert_equivalent_repr(cirq_google.PhysicalZTag(), setup_code='import cirq\nimport cirq_google\n')