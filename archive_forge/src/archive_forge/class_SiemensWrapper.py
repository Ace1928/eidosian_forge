import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
class SiemensWrapper(Wrapper):
    """Wrapper for Siemens format DICOMs

    Adds attributes:

    * csa_header : mapping
    * b_matrix : (3,3) array
    * q_vector : (3,) array
    """
    is_csa = True

    def __init__(self, dcm_data, csa_header=None):
        """Initialize Siemens wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`.

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  If `csa_header`
           is None, it should also be possible to extract a CSA header from
           `dcm_data`. Usually this will be a ``dicom.dataset.Dataset`` object
           resulting from reading a DICOM file.  A dict should also work.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.  If
           None, we try and read the CSA information from `dcm_data`.
           If this fails, we fall back to an empty dict.
        """
        super().__init__(dcm_data)
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data
        if csa_header is None:
            csa_header = csar.get_csa_header(dcm_data)
            if csa_header is None:
                csa_header = {}
        self.csa_header = csa_header

    @one_time
    def slice_normal(self):
        std_slice_normal = super().slice_normal
        csa_slice_normal = csar.get_slice_normal(self.csa_header)
        if std_slice_normal is None and csa_slice_normal is None:
            return None
        elif std_slice_normal is None:
            return np.array(csa_slice_normal)
        elif csa_slice_normal is None:
            return std_slice_normal
        else:
            dot_prod = np.dot(csa_slice_normal, std_slice_normal)
            assert np.allclose(np.fabs(dot_prod), 1.0, atol=1e-05)
            if dot_prod < 0:
                return -std_slice_normal
            else:
                return std_slice_normal

    @one_time
    def series_signature(self):
        """Add ICE dims from CSA header to signature"""
        signature = super().series_signature
        ice = csar.get_ice_dims(self.csa_header)
        if ice is not None:
            ice = ice[:6] + ice[8:9]
        signature['ICE_Dims'] = (ice, operator.eq)
        return signature

    @one_time
    def b_matrix(self):
        """Get DWI B matrix referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        B : (3,3) array or None
           B matrix in *voxel* orientation space.  Returns None if this is
           not a Siemens header with the required information.  We return
           None if this is a b0 acquisition
        """
        hdr = self.csa_header
        B = csar.get_b_matrix(hdr)
        if B is None:
            bval_requested = csar.get_b_value(hdr)
            if bval_requested is None:
                return None
            if bval_requested != 0:
                raise csar.CSAError('No B matrix and b value != 0')
            return np.zeros((3, 3))
        R = self.rotation_matrix.T
        B_vox = np.dot(R, np.dot(B, R.T))
        return nearest_pos_semi_def(B_vox)

    @one_time
    def q_vector(self):
        """Get DWI q vector referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        q: (3,) array
           Estimated DWI q vector in *voxel* orientation space.  Returns
           None if this is not (detectably) a DWI
        """
        B = self.b_matrix
        if B is None:
            return None
        return B2q(B, tol=1e-08)