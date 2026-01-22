import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class OutlierCount(CommandLine):
    """Calculates number of 'outliers' at each time point of a
    a 3D+time dataset.

    For complete details, see the `3dToutcount Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dToutcount.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> toutcount = afni.OutlierCount()
    >>> toutcount.inputs.in_file = 'functional.nii'
    >>> toutcount.cmdline  # doctest: +ELLIPSIS
    '3dToutcount -qthr 0.00100 functional.nii'
    >>> res = toutcount.run()  # doctest: +SKIP

    """
    _cmd = '3dToutcount'
    input_spec = OutlierCountInputSpec
    output_spec = OutlierCountOutputSpec
    _terminal_output = 'file_split'

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if self.terminal_output == 'none':
            self.terminal_output = 'file_split'
        if not self.inputs.save_outliers:
            skip += ['outliers_file']
        return super(OutlierCount, self)._parse_inputs(skip)

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(OutlierCount, self)._run_interface(runtime, correct_return_codes)
        with open(op.abspath(self.inputs.out_file), 'w') as outfh:
            outfh.write(runtime.stdout or runtime.merged)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        if self.inputs.save_outliers:
            outputs['out_outliers'] = op.abspath(self.inputs.outliers_file)
        return outputs