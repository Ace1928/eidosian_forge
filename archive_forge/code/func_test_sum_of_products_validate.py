import cirq
import pytest
def test_sum_of_products_validate():
    control_val = cirq.SumOfProducts(((1, 2), (0, 1)))
    _ = control_val.validate([2, 3])
    with pytest.raises(ValueError):
        _ = control_val.validate([2, 2])
    with pytest.raises(ValueError):
        _ = control_val.validate([2])