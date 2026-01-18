from thinc.api import (
def test_warmup_linear():
    rates = warmup_linear(1.0, 2, 10)
    expected = [0.0, 0.5, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    for i in range(11):
        assert next(rates) == expected[i]