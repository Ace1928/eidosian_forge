import cirq
import pytest
def test_sum_of_products_str():
    c = cirq.SumOfProducts(((1, 0), (0, 1)))
    assert str(c) == 'C_01_10'
    c = cirq.SumOfProducts(((1, 0), (0, 1)), name='xor')
    assert str(c) == 'C_xor'