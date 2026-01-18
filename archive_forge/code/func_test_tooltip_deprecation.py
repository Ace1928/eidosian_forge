import inspect
import pytest
from ..widget_string import Combobox, Text
def test_tooltip_deprecation():
    caller_path = inspect.stack(context=0)[1].filename
    with pytest.deprecated_call() as record:
        w = Text(description_tooltip='testing')
    assert len(record) == 1
    assert record[0].filename == caller_path
    with pytest.deprecated_call() as record:
        w.description_tooltip
    assert len(record) == 1
    assert record[0].filename == caller_path
    with pytest.deprecated_call() as record:
        w.description_tooltip == 'testing'
    assert len(record) == 1
    assert record[0].filename == caller_path
    with pytest.deprecated_call() as record:
        w.description_tooltip = 'second value'
    assert len(record) == 1
    assert record[0].filename == caller_path
    assert w.tooltip == 'second value'