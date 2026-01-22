import numpy as np
from . import spm99analyze as spm99  # module import
class Spm2AnalyzeHeader(spm99.Spm99AnalyzeHeader):
    """Class for SPM2 variant of basic Analyze header

    SPM2 variant adds the following to basic Analyze format:

    * voxel origin;
    * slope scaling of data;
    * reading - but not writing - intercept of data.
    """
    template_dtype = header_dtype

    def get_slope_inter(self):
        """Get data scaling (slope) and intercept from header data

        Uses the algorithm from SPM2 spm_vol_ana.m by John Ashburner

        Parameters
        ----------
        self : header
           Mapping with fields:
           * scl_slope - slope
           * scl_inter - possible intercept (SPM2 use - shared by nifti)
           * glmax - the (recorded) maximum value in the data (unscaled)
           * glmin - recorded minimum unscaled value
           * cal_max - the calibrated (scaled) maximum value in the dataset
           * cal_min - ditto minimum value

        Returns
        -------
        scl_slope : None or float
            slope.  None if there is no valid scaling from these fields
        scl_inter : None or float
            intercept.  Also None if there is no valid slope, intercept

        Examples
        --------
        >>> fields = {'scl_slope': 1, 'scl_inter': 0, 'glmax': 0, 'glmin': 0,
        ...           'cal_max': 0, 'cal_min': 0}
        >>> hdr = Spm2AnalyzeHeader()
        >>> for key, value in fields.items():
        ...     hdr[key] = value
        >>> hdr.get_slope_inter()
        (1.0, 0.0)
        >>> hdr['scl_inter'] = 0.5
        >>> hdr.get_slope_inter()
        (1.0, 0.5)
        >>> hdr['scl_inter'] = np.nan
        >>> hdr.get_slope_inter()
        (1.0, 0.0)

        If 'scl_slope' is 0, nan or inf, cannot use 'scl_slope'.
        Without valid information in the gl / cal fields, we cannot get
        scaling, and return None

        >>> hdr['scl_slope'] = 0
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['scl_slope'] = np.nan
        >>> hdr.get_slope_inter()
        (None, None)

        Valid information in the gl AND cal fields are needed

        >>> hdr['cal_max'] = 0.8
        >>> hdr['cal_min'] = 0.2
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['glmax'] = 110
        >>> hdr['glmin'] = 10
        >>> np.allclose(hdr.get_slope_inter(), [0.6/100, 0.2-0.6/100*10])
        True
        """
        slope = float(self['scl_slope'])
        if np.isfinite(slope) and slope:
            inter = float(self['scl_inter'])
            if not np.isfinite(inter):
                inter = 0.0
            return (slope, inter)
        unscaled_range = self['glmax'] - self['glmin']
        scaled_range = self['cal_max'] - self['cal_min']
        if unscaled_range and scaled_range:
            slope = float(scaled_range) / unscaled_range
            inter = self['cal_min'] - slope * self['glmin']
            return (slope, inter)
        return (None, None)

    @classmethod
    def may_contain_header(klass, binaryblock):
        if len(binaryblock) < klass.sizeof_hdr:
            return False
        hdr_struct = np.ndarray(shape=(), dtype=header_dtype, buffer=binaryblock[:klass.sizeof_hdr])
        bs_hdr_struct = hdr_struct.byteswap()
        return binaryblock[344:348] not in (b'ni1\x00', b'n+1\x00') and 348 in (hdr_struct['sizeof_hdr'], bs_hdr_struct['sizeof_hdr'])