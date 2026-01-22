import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AutoTLRC(AFNICommand):
    """A minimal wrapper for the AutoTLRC script
    The only option currently supported is no_ss.
    For complete details, see the `3dQwarp Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/@auto_tlrc.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> autoTLRC = afni.AutoTLRC()
    >>> autoTLRC.inputs.in_file = 'structural.nii'
    >>> autoTLRC.inputs.no_ss = True
    >>> autoTLRC.inputs.base = "TT_N27+tlrc"
    >>> autoTLRC.cmdline
    '@auto_tlrc -base TT_N27+tlrc -input structural.nii -no_ss'
    >>> res = autoTLRC.run()  # doctest: +SKIP

    """
    _cmd = '@auto_tlrc'
    input_spec = AutoTLRCInputSpec
    output_spec = AFNICommandOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        ext = '.HEAD'
        outputs['out_file'] = os.path.abspath(self._gen_fname(self.inputs.in_file, suffix='+tlrc') + ext)
        return outputs