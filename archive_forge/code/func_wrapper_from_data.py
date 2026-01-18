import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
def wrapper_from_data(dcm_data):
    """Create DICOM wrapper from DICOM data object

    Parameters
    ----------
    dcm_data : ``dicom.dataset.Dataset`` instance or similar
       Object allowing attribute access, with DICOM attributes.
       Probably a dataset as read by ``pydicom``.

    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    """
    sop_class = dcm_data.get('SOPClassUID')
    if sop_class == '1.2.840.10008.5.1.4.1.1.4.1':
        return MultiframeWrapper(dcm_data)
    try:
        csa = csar.get_csa_header(dcm_data)
    except csar.CSAReadError as e:
        warnings.warn(f'Error while attempting to read CSA header: {e.args}\nIgnoring Siemens private (CSA) header info.')
        csa = None
    if csa is None:
        return Wrapper(dcm_data)
    if csar.is_mosaic(csa):
        return MosaicWrapper(dcm_data, csa)
    return SiemensWrapper(dcm_data, csa)