import numpy as np
from .analyze import AnalyzeHeader
from .batteryrunners import Report
from .filebasedimages import ImageFileError
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .spatialimages import HeaderDataError
class Nifti2Image(Nifti1Image):
    """Class for single file NIfTI2 format image"""
    header_class = Nifti2Header
    _meta_sniff_len = header_class.sizeof_hdr