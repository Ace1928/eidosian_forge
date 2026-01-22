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
def alternative_hillshade(azimuth, elev, z):
    illum = _sph2cart(*_azimuth2math(azimuth, elev))
    illum = np.array(illum)
    dy, dx = np.gradient(-z)
    dy = -dy
    dz = np.ones_like(dy)
    normals = np.dstack([dx, dy, dz])
    normals /= np.linalg.norm(normals, axis=2)[..., None]
    intensity = np.tensordot(normals, illum, axes=(2, 0))
    intensity -= intensity.min()
    intensity /= np.ptp(intensity)
    return intensity