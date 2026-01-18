import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_set_alpha():
    art = martist.Artist()
    with pytest.raises(TypeError, match='^alpha must be numeric or None'):
        art.set_alpha('string')
    with pytest.raises(TypeError, match='^alpha must be numeric or None'):
        art.set_alpha([1, 2, 3])
    with pytest.raises(ValueError, match='outside 0-1 range'):
        art.set_alpha(1.1)
    with pytest.raises(ValueError, match='outside 0-1 range'):
        art.set_alpha(np.nan)