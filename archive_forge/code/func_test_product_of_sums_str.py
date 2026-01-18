import cirq
import pytest
def test_product_of_sums_str():
    c = cirq.ProductOfSums([(0, 1), 1, 0, (0, 2)])
    assert str(c) == 'C01C1C0C02'