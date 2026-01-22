import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
class EcatHeader(WrapStruct, SpatialHeader):
    """Class for basic Ecat PET header

    Sub-parts of standard Ecat File

    * main header
    * matrix list
      which lists the information for each frame collected (can have 1 to many
      frames)
    * subheaders specific to each frame with possibly-variable sized data
      blocks

    This just reads the main Ecat Header, it does not load the data or read the
    mlist or any sub headers
    """
    template_dtype = hdr_dtype
    _ft_codes = file_type_codes
    _patient_orient_codes = patient_orient_codes

    def __init__(self, binaryblock=None, endianness=None, check=True):
        """Initialize Ecat header from bytes object

        Parameters
        ----------
        binaryblock : {None, bytes} optional
            binary block to set into header, By default, None in which case we
            insert default empty header block
        endianness : {None, '<', '>', other endian code}, optional
            endian code of binary block, If None, guess endianness
            from the data
        check : {True, False}, optional
            Whether to check and fix header for errors.  No checks currently
            implemented, so value has no effect.
        """
        super().__init__(binaryblock, endianness, check)

    @classmethod
    def guessed_endian(klass, hdr):
        """Guess endian from MAGIC NUMBER value of header data"""
        if not hdr['sw_version'] == 74:
            return swapped_code
        else:
            return native_code

    @classmethod
    def default_structarr(klass, endianness=None):
        """Return header data for empty header with given endianness"""
        hdr_data = super().default_structarr(endianness)
        hdr_data['magic_number'] = 'MATRIX72'
        hdr_data['sw_version'] = 74
        hdr_data['num_frames'] = 0
        hdr_data['file_type'] = 0
        hdr_data['ecat_calibration_factor'] = 1.0
        return hdr_data

    def get_data_dtype(self):
        """Get numpy dtype for data from header"""
        raise NotImplementedError('dtype is only valid from subheaders')

    def get_patient_orient(self):
        """gets orientation of patient based on code stored
        in header, not always reliable
        """
        code = self._structarr['patient_orientation'].item()
        if code not in self._patient_orient_codes:
            raise KeyError('Ecat Orientation CODE %d not recognized' % code)
        return self._patient_orient_codes[code]

    def get_filetype(self):
        """Type of ECAT Matrix File from code stored in header"""
        code = self._structarr['file_type'].item()
        if code not in self._ft_codes:
            raise KeyError('Ecat Filetype CODE %d not recognized' % code)
        return self._ft_codes[code]

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return ()