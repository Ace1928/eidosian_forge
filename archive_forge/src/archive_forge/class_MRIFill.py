import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIFill(FSCommand):
    """
    This program creates hemispheric cutting planes and fills white matter
    with specific values for subsequent surface tessellation.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MRIFill
    >>> fill = MRIFill()
    >>> fill.inputs.in_file = 'wm.mgz' # doctest: +SKIP
    >>> fill.inputs.out_file = 'filled.mgz' # doctest: +SKIP
    >>> fill.cmdline # doctest: +SKIP
    'mri_fill wm.mgz filled.mgz'
    """
    _cmd = 'mri_fill'
    input_spec = MRIFillInputSpec
    output_spec = MRIFillOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        if isdefined(self.inputs.log_file):
            outputs['log_file'] = os.path.abspath(self.inputs.log_file)
        return outputs