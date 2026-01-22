import os
import re as regex
from ..base import (
class Cerebro(CommandLine):
    """
    Cerebrum/cerebellum labeling tool
    This program performs automated labeling of cerebellum and cerebrum in T1 MRI.
    Input MRI should be skull-stripped or a brain-only mask should be provided.


    http://brainsuite.org/processing/surfaceextraction/cerebrum/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> cerebro = brainsuite.Cerebro()
    >>> cerebro.inputs.inputMRIFile = example_data('structural.nii')
    >>> cerebro.inputs.inputAtlasMRIFile = 'atlasMRIVolume.img'
    >>> cerebro.inputs.inputAtlasLabelFile = 'atlasLabels.img'
    >>> cerebro.inputs.inputBrainMaskFile = example_data('mask.nii')
    >>> results = cerebro.run() #doctest: +SKIP

    """
    input_spec = CerebroInputSpec
    output_spec = CerebroOutputSpec
    _cmd = 'cerebro'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        fileToSuffixMap = {'outputCerebrumMaskFile': '.cerebrum.mask.nii.gz', 'outputLabelVolumeFile': '.hemi.label.nii.gz', 'outputWarpTransformFile': '.warp', 'outputAffineTransformFile': '.air'}
        if name in fileToSuffixMap:
            return getFileName(self.inputs.inputMRIFile, fileToSuffixMap[name])
        return None

    def _list_outputs(self):
        return l_outputs(self)