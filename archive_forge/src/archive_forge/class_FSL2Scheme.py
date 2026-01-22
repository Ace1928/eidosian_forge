import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class FSL2Scheme(StdOutCommandLine):
    """
    Converts b-vectors and b-values from FSL format to a Camino scheme file.

    Examples
    --------

    >>> import nipype.interfaces.camino as cmon
    >>> makescheme = cmon.FSL2Scheme()
    >>> makescheme.inputs.bvec_file = 'bvecs'
    >>> makescheme.inputs.bvec_file = 'bvals'
    >>> makescheme.run()                  # doctest: +SKIP

    """
    _cmd = 'fsl2scheme'
    input_spec = FSL2SchemeInputSpec
    output_spec = FSL2SchemeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['scheme'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.bvec_file)
        return name + '.scheme'