import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
def test_setting_backend_without_plot_raises(monkeypatch):
    module = types.ModuleType('pandas_plot_backend')
    monkeypatch.setitem(sys.modules, 'pandas_plot_backend', module)
    assert pandas.options.plotting.backend == 'matplotlib'
    with pytest.raises(ValueError, match="Could not find plotting backend 'pandas_plot_backend'."):
        pandas.set_option('plotting.backend', 'pandas_plot_backend')
    assert pandas.options.plotting.backend == 'matplotlib'