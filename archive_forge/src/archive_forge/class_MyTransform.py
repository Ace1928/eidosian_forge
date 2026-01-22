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
class MyTransform(Transform):
    input_dims = output_dims = 2

    def __init__(self, resolution):
        """
            Resolution is the number of steps to interpolate between each input
            line segment to approximate its path in transformed space.
            """
        Transform.__init__(self)
        self._resolution = resolution

    def transform(self, ll):
        x, y = ll.T
        return np.column_stack([x, y - x])
    transform_non_affine = transform

    def transform_path(self, path):
        ipath = path.interpolated(self._resolution)
        return Path(self.transform(ipath.vertices), ipath.codes)
    transform_path_non_affine = transform_path

    def inverted(self):
        return MyTransformInv(self._resolution)