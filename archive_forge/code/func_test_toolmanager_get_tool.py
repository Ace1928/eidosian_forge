import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_toolmanager_get_tool():
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.gcf()
    rubberband = fig.canvas.manager.toolmanager.get_tool('rubberband')
    assert isinstance(rubberband, RubberbandBase)
    assert fig.canvas.manager.toolmanager.get_tool(rubberband) is rubberband
    with pytest.warns(UserWarning, match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.get_tool('foo') is None
    assert fig.canvas.manager.toolmanager.get_tool('foo', warn=False) is None
    with pytest.warns(UserWarning, match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.trigger_tool('foo') is None