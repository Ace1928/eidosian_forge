from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
class Nifti1Image(Nifti1Pair, SerializableImage):
    """Class for single file NIfTI1 format image"""
    header_class = Nifti1Header
    valid_exts = ('.nii',)
    files_types = (('image', '.nii'),)

    @staticmethod
    def _get_fileholders(file_map):
        """Return fileholder for header and image

        For single-file niftis, the fileholder for the header and the image
        will be the same
        """
        return (file_map['image'], file_map['image'])

    def update_header(self):
        """Harmonize header with image data and affine"""
        super().update_header()
        hdr = self._header
        hdr['magic'] = hdr.single_magic