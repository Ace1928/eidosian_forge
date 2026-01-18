from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_set_offset_transform():
    skew = mtransforms.Affine2D().skew(2, 2)
    init = mcollections.Collection(offset_transform=skew)
    late = mcollections.Collection()
    late.set_offset_transform(skew)
    assert skew == init.get_offset_transform() == late.get_offset_transform()