import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Fourier(AFNICommand):
    """Program to lowpass and/or highpass each voxel time series in a
    dataset, via the FFT

    For complete details, see the `3dFourier Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dFourier.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> fourier = afni.Fourier()
    >>> fourier.inputs.in_file = 'functional.nii'
    >>> fourier.inputs.retrend = True
    >>> fourier.inputs.highpass = 0.005
    >>> fourier.inputs.lowpass = 0.1
    >>> fourier.cmdline
    '3dFourier -highpass 0.005000 -lowpass 0.100000 -prefix functional_fourier -retrend functional.nii'
    >>> res = fourier.run()  # doctest: +SKIP

    """
    _cmd = '3dFourier'
    input_spec = FourierInputSpec
    output_spec = AFNICommandOutputSpec