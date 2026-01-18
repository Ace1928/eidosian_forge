import cirq
import pytest
@pytest.mark.parametrize('data', [((1,),), ((0, 1), (1,)), [(0, 1), (1, 0)]])
def test_product_of_sums_repr(data):
    cirq.testing.assert_equivalent_repr(cirq.ProductOfSums(data))