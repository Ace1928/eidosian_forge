import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Bandpass(AFNICommand):
    """Program to lowpass and/or highpass each voxel time series in a
    dataset, offering more/different options than Fourier

    For complete details, see the `3dBandpass Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dBandpass.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> from nipype.testing import  example_data
    >>> bandpass = afni.Bandpass()
    >>> bandpass.inputs.in_file = 'functional.nii'
    >>> bandpass.inputs.highpass = 0.005
    >>> bandpass.inputs.lowpass = 0.1
    >>> bandpass.cmdline
    '3dBandpass -prefix functional_bp 0.005000 0.100000 functional.nii'
    >>> res = bandpass.run()  # doctest: +SKIP

    """
    _cmd = '3dBandpass'
    input_spec = BandpassInputSpec
    output_spec = AFNICommandOutputSpec