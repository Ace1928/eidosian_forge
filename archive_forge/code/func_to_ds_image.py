import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
def to_ds_image(binned, rgba):
    if binned.ndim == 2:
        return tf.Image(uint8_to_uint32(rgba), coords=binned.coords, dims=binned.dims)
    elif binned.ndim == 3:
        return tf.Image(uint8_to_uint32(rgba), dims=binned.dims[:-1], coords=dict([(binned.dims[1], binned.coords[binned.dims[1]]), (binned.dims[0], binned.coords[binned.dims[0]])]))
    else:
        raise ValueError('Aggregate must be 2D or 3D.')