import os
import param
import numpy as np
import xarray as xr
from holoviews.core.util import get_param_values
from holoviews.core.data import XArrayInterface
from holoviews.element import Image as HvImage, QuadMesh as HvQuadMesh
from holoviews.operation.datashader import regrid
from ..element import Image, QuadMesh, is_geographic

        Cleans existing weight files.
        