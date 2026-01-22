from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
@staticmethod
def _get_fileholders(file_map):
    """Return fileholder for header and image

        Allows single-file image types to return one fileholder for both types.
        For Analyze there are two fileholders, one for the header, one for the
        image.
        """
    return (file_map['header'], file_map['image'])