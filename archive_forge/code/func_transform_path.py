import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, Transform
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
def transform_path(self, path):
    ipath = path.interpolated(self._resolution)
    return Path(self.transform(ipath.vertices), ipath.codes)