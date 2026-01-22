import plotly.graph_objs as go
from _plotly_utils.basevalidators import ColorscaleValidator
from ._core import apply_default_cascade, init_figure, configure_animation_controls
from .imshow_utils import rescale_intensity, _integer_ranges, _integer_types
import pandas as pd
import numpy as np
import itertools
from plotly.utils import image_array_to_data_uri
def _infer_zmax_from_type(img):
    dt = img.dtype.type
    rtol = 1.05
    if dt in _integer_types:
        return _integer_ranges[dt][1]
    else:
        im_max = img[np.isfinite(img)].max()
        if im_max <= 1 * rtol:
            return 1
        elif im_max <= 255 * rtol:
            return 255
        elif im_max <= 65535 * rtol:
            return 65535
        else:
            return 2 ** 32