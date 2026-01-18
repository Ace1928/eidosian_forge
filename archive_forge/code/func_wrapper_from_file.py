import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
def wrapper_from_file(file_like, *args, **kwargs):
    """Create DICOM wrapper from `file_like` object

    Parameters
    ----------
    file_like : object
       filename string or file-like object, pointing to a valid DICOM
       file readable by ``pydicom``
    \\*args : positional
        args to ``dicom.dcmread`` command.
    \\*\\*kwargs : keyword
        args to ``dicom.dcmread`` command.  ``force=True`` might be a
        likely keyword argument.

    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    """
    with ImageOpener(file_like) as fobj:
        dcm_data = pydicom.dcmread(fobj, *args, **kwargs)
    return wrapper_from_data(dcm_data)