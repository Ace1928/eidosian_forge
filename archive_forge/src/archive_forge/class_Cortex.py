import os
import re as regex
from ..base import (
class Cortex(CommandLine):
    """
    cortex extractor
    This program produces a cortical mask using tissue fraction estimates
    and a co-registered cerebellum/hemisphere mask.

    http://brainsuite.org/processing/surfaceextraction/cortex/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> cortex = brainsuite.Cortex()
    >>> cortex.inputs.inputHemisphereLabelFile = example_data('mask.nii')
    >>> cortex.inputs.inputTissueFractionFile = example_data('tissues.nii.gz')
    >>> results = cortex.run() #doctest: +SKIP

    """
    input_spec = CortexInputSpec
    output_spec = CortexOutputSpec
    _cmd = 'cortex'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputCerebrumMask':
            return getFileName(self.inputs.inputHemisphereLabelFile, '.init.cortex.mask.nii.gz')
        return None

    def _list_outputs(self):
        return l_outputs(self)