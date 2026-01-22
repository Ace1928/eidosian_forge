import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def _transform_path(self, subslice=None):
    """
        Put a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
    if subslice is not None:
        xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
        _path = Path(np.asarray(xy).T, _interpolation_steps=self._path._interpolation_steps)
    else:
        _path = self._path
    self._transformed_path = TransformedPath(_path, self.get_transform())