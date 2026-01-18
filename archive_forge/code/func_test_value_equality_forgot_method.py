import pytest
import cirq
def test_value_equality_forgot_method():
    with pytest.raises(TypeError, match='_value_equality_values_'):

        @cirq.value_equality
        class _:
            pass