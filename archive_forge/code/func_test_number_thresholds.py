import pytest
from panel.widgets.indicators import (
def test_number_thresholds(document, comm):
    number = Number(value=0, colors=[(0.33, 'green'), (0.66, 'yellow'), (1, 'red')])
    model = number.get_root(document, comm)
    assert 'green' in model.text
    number.value = 0.5
    assert 'yellow' in model.text
    number.value = 0.7
    assert 'red' in model.text