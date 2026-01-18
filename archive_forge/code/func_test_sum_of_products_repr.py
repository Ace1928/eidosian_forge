import cirq
import pytest
@pytest.mark.parametrize('data', [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))])
def test_sum_of_products_repr(data):
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data))
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data, name='CustomName'))