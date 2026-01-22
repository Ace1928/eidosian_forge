import os
import re as regex
from ..base import (
class Pialmesh(CommandLine):
    """
    pialmesh
    computes a pial surface model using an inner WM/GM mesh and a tissue fraction map.

    http://brainsuite.org/processing/surfaceextraction/pial/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> pialmesh = brainsuite.Pialmesh()
    >>> pialmesh.inputs.inputSurfaceFile = 'input_mesh.dfs'
    >>> pialmesh.inputs.inputTissueFractionFile = 'frac_file.nii.gz'
    >>> pialmesh.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = pialmesh.run() #doctest: +SKIP

    """
    input_spec = PialmeshInputSpec
    output_spec = PialmeshOutputSpec
    _cmd = 'pialmesh'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputSurfaceFile':
            return getFileName(self.inputs.inputSurfaceFile, '.pial.cortex.dfs')
        return None

    def _list_outputs(self):
        return l_outputs(self)