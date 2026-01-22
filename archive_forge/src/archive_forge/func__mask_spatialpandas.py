import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _mask_spatialpandas(masked_xvals, masked_yvals, geometry):
    from spatialpandas.geometry import PointArray, Polygon
    points = PointArray((masked_xvals.astype('float'), masked_yvals.astype('float')))
    poly = Polygon([np.concatenate([geometry, geometry[:1]]).flatten()])
    return points.intersects(poly)