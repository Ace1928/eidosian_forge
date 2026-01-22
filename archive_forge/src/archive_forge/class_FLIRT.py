import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FLIRT(FSLCommand):
    """FSL FLIRT wrapper for coregistration

    For complete details, see the `FLIRT Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT>`_

    To print out the command line help, use:
        fsl.FLIRT().inputs_help()

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import example_data
    >>> flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    >>> flt.inputs.in_file = 'structural.nii'
    >>> flt.inputs.reference = 'mni.nii'
    >>> flt.inputs.output_type = "NIFTI_GZ"
    >>> flt.cmdline # doctest: +ELLIPSIS
    'flirt -in structural.nii -ref mni.nii -out structural_flirt.nii.gz -omat structural_flirt.mat -bins 640 -searchcost mutualinfo'
    >>> res = flt.run() #doctest: +SKIP

    """
    _cmd = 'flirt'
    input_spec = FLIRTInputSpec
    output_spec = FLIRTOutputSpec
    _log_written = False

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(FLIRT, self).aggregate_outputs(runtime=runtime, needed_outputs=needed_outputs)
        if self.inputs.save_log and (not self._log_written):
            with open(outputs.out_log, 'a') as text_file:
                text_file.write(runtime.stdout + '\n')
            self._log_written = True
        return outputs

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if self.inputs.save_log and (not self.inputs.verbose):
            self.inputs.verbose = 1
        if self.inputs.apply_xfm and (not (self.inputs.in_matrix_file or self.inputs.uses_qform)):
            raise RuntimeError('Argument apply_xfm requires in_matrix_file or uses_qform arguments to run')
        skip.append('save_log')
        return super(FLIRT, self)._parse_inputs(skip=skip)