import numpy as np
from .analyze import AnalyzeHeader
from .batteryrunners import Report
from .filebasedimages import ImageFileError
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .spatialimages import HeaderDataError
class Nifti2Header(Nifti1Header):
    """Class for NIfTI2 header

    NIfTI2 is a slightly simplified variant of NIfTI1 which replaces 32-bit
    floats with 64-bit floats, and increases some integer widths to 32 or 64
    bits.
    """
    template_dtype = header_dtype
    pair_vox_offset = 0
    single_vox_offset = 544
    pair_magic = b'ni2'
    single_magic = b'n+2'
    sizeof_hdr = 540
    quaternion_threshold = -np.finfo(np.float64).eps * 3

    def get_data_shape(self):
        """Get shape of data

        Examples
        --------
        >>> hdr = Nifti2Header()
        >>> hdr.get_data_shape()
        (0,)
        >>> hdr.set_data_shape((1,2,3))
        >>> hdr.get_data_shape()
        (1, 2, 3)

        Expanding number of dimensions gets default zooms

        >>> hdr.get_zooms()
        (1.0, 1.0, 1.0)

        Notes
        -----
        Does not use Nifti1 freesurfer hack for large vectors described in
        :meth:`Nifti1Header.set_data_shape`
        """
        return AnalyzeHeader.get_data_shape(self)

    def set_data_shape(self, shape):
        """Set shape of data

        If ``ndims == len(shape)`` then we set zooms for dimensions higher than
        ``ndims`` to 1.0

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape

        Notes
        -----
        Does not apply nifti1 Freesurfer hack for long vectors (see
        :meth:`Nifti1Header.set_data_shape`)
        """
        AnalyzeHeader.set_data_shape(self, shape)

    @classmethod
    def default_structarr(klass, endianness=None):
        """Create empty header binary block with given endianness"""
        hdr_data = super().default_structarr(endianness)
        hdr_data['eol_check'] = (13, 10, 26, 10)
        return hdr_data
    ' Checks only below here '

    @classmethod
    def _get_checks(klass):
        return super()._get_checks() + (klass._chk_eol_check,)

    @staticmethod
    def _chk_eol_check(hdr, fix=False):
        rep = Report(HeaderDataError)
        if np.all(hdr['eol_check'] == (13, 10, 26, 10)):
            return (hdr, rep)
        if np.all(hdr['eol_check'] == 0):
            rep.problem_level = 20
            rep.problem_msg = 'EOL check all 0'
            if fix:
                hdr['eol_check'] = (13, 10, 26, 10)
                rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
            return (hdr, rep)
        rep.problem_level = 40
        rep.problem_msg = 'EOL check not 0 or 13, 10, 26, 10; data may be corrupted by EOL conversion'
        if fix:
            hdr['eol_check'] = (13, 10, 26, 10)
            rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
        return (hdr, rep)

    @classmethod
    def may_contain_header(klass, binaryblock):
        if len(binaryblock) < klass.sizeof_hdr:
            return False
        hdr_struct = np.ndarray(shape=(), dtype=header_dtype, buffer=binaryblock[:klass.sizeof_hdr])
        bs_hdr_struct = hdr_struct.byteswap()
        return 540 in (hdr_struct['sizeof_hdr'], bs_hdr_struct['sizeof_hdr'])