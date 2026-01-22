import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TProject(AFNICommand):
    """
    This program projects (detrends) out various 'nuisance' time series from
    each voxel in the input dataset.  Note that all the projections are done
    via linear regression, including the frequency-based options such
    as ``-passband``.  In this way, you can bandpass time-censored data, and at
    the same time, remove other time series of no interest
    (e.g., physiological estimates, motion parameters).
    Shifts voxel time series from input so that separate slices are aligned to
    the same temporal origin.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tproject = afni.TProject()
    >>> tproject.inputs.in_file = 'functional.nii'
    >>> tproject.inputs.bandpass = (0.00667, 99999)
    >>> tproject.inputs.polort = 3
    >>> tproject.inputs.automask = True
    >>> tproject.inputs.out_file = 'projected.nii.gz'
    >>> tproject.cmdline
    '3dTproject -input functional.nii -automask -bandpass 0.00667 99999 -polort 3 -prefix projected.nii.gz'
    >>> res = tproject.run()  # doctest: +SKIP

    See Also
    --------
    For complete details, see the `3dTproject Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTproject.html>`__

    """
    _cmd = '3dTproject'
    input_spec = TProjectInputSpec
    output_spec = AFNICommandOutputSpec