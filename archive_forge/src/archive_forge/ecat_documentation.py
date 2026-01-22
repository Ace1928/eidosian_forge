import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
Write ECAT7 image to `file_map` or contained ``self.file_map``

        The format consist of:

        - A main header (512L) with dictionary entries in the form
            [numAvail, nextDir, previousDir, numUsed]
        - For every frame (3D volume in 4D data)
          - A subheader (size = frame_offset)
          - Frame data (3D volume)
        