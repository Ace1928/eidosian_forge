import os
import re as regex
from ..base import (
class Bse(CommandLine):
    """
    brain surface extractor (BSE)
    This program performs automated skull and scalp removal on T1-weighted MRI volumes.

    http://brainsuite.org/processing/surfaceextraction/bse/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> bse = brainsuite.Bse()
    >>> bse.inputs.inputMRIFile = example_data('structural.nii')
    >>> results = bse.run() #doctest: +SKIP

    """
    input_spec = BseInputSpec
    output_spec = BseOutputSpec
    _cmd = 'bse'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        fileToSuffixMap = {'outputMRIVolume': '.bse.nii.gz', 'outputMaskFile': '.mask.nii.gz'}
        if name in fileToSuffixMap:
            return getFileName(self.inputs.inputMRIFile, fileToSuffixMap[name])
        return None

    def _list_outputs(self):
        return l_outputs(self)