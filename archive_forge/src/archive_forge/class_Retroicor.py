import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Retroicor(AFNICommand):
    """Performs Retrospective Image Correction for physiological
    motion effects, using a slightly modified version of the
    RETROICOR algorithm

    The durations of the physiological inputs are assumed to equal
    the duration of the dataset. Any constant sampling rate may be
    used, but 40 Hz seems to be acceptable. This program's cardiac
    peak detection algorithm is rather simplistic, so you might try
    using the scanner's cardiac gating output (transform it to a
    spike wave if necessary).

    This program uses slice timing information embedded in the
    dataset to estimate the proper cardiac/respiratory phase for
    each slice. It makes sense to run this program before any
    program that may destroy the slice timings (e.g. 3dvolreg for
    motion correction).

    For complete details, see the `3dretroicor Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dretroicor.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> ret = afni.Retroicor()
    >>> ret.inputs.in_file = 'functional.nii'
    >>> ret.inputs.card = 'mask.1D'
    >>> ret.inputs.resp = 'resp.1D'
    >>> ret.inputs.outputtype = 'NIFTI'
    >>> ret.cmdline
    '3dretroicor -prefix functional_retroicor.nii -resp resp.1D -card mask.1D functional.nii'
    >>> res = ret.run()  # doctest: +SKIP

    """
    _cmd = '3dretroicor'
    input_spec = RetroicorInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file':
            if not isdefined(self.inputs.card) and (not isdefined(self.inputs.resp)):
                return None
        return super(Retroicor, self)._format_arg(name, trait_spec, value)