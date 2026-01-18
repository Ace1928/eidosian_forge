import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_register_cmap():
    new_cm = mpl.colormaps['viridis']
    target = 'viridis2'
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\.register\\(name\\)'):
        cm.register_cmap(target, new_cm)
    assert mpl.colormaps[target] == new_cm
    with pytest.raises(ValueError, match='Arguments must include a name or a Colormap'):
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\.register\\(name\\)'):
            cm.register_cmap()
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\.unregister\\(name\\)'):
        cm.unregister_cmap(target)
    with pytest.raises(ValueError, match=f'{target!r} is not a valid value for name;'):
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\[name\\]'):
            cm.get_cmap(target)
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\.unregister\\(name\\)'):
        cm.unregister_cmap(target)
    with pytest.raises(TypeError, match="'cmap' must be"):
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='matplotlib\\.colormaps\\.register\\(name\\)'):
            cm.register_cmap('nome', cmap='not a cmap')