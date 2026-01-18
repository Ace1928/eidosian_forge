from contextlib import contextmanager
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_use_priority():
    manage_plugins.use_plugin(priority_plugin)
    plug, func = manage_plugins.plugin_store['imread'][0]
    np.testing.assert_equal(plug, priority_plugin)
    manage_plugins.use_plugin('matplotlib')
    plug, func = manage_plugins.plugin_store['imread'][0]
    np.testing.assert_equal(plug, 'matplotlib')