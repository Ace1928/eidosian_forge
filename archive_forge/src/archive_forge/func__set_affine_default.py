from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
def _set_affine_default(self):
    """If goodRASFlag is 0, set the default affine"""
    self._structarr['goodRASFlag'] = 1
    self._structarr['delta'] = 1
    self._structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
    self._structarr['Pxyz_c'] = 0