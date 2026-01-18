import pytest
import numpy as np
import cirq
@pytest.mark.parametrize('val,mixture', ((ReturnsValidTuple(), ((0.4, 'a'), (0.6, 'b'))), (ReturnsNonnormalizedTuple(), ((0.4, 'a'), (0.4, 'b'))), (ReturnsUnitary(), ((1.0, np.ones((2, 2))),))))
def test_objects_with_mixture(val, mixture):
    expected_keys, expected_values = zip(*mixture)
    keys, values = zip(*cirq.mixture(val))
    np.testing.assert_almost_equal(keys, expected_keys)
    np.testing.assert_equal(values, expected_values)
    keys, values = zip(*cirq.mixture(val, ((0.3, 'a'), (0.7, 'b'))))
    np.testing.assert_almost_equal(keys, expected_keys)
    np.testing.assert_equal(values, expected_values)