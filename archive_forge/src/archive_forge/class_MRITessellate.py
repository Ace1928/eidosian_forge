import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRITessellate(FSCommand):
    """
    Uses Freesurfer's mri_tessellate to create surfaces by tessellating a given input volume

    Example
    -------

    >>> import nipype.interfaces.freesurfer as fs
    >>> tess = fs.MRITessellate()
    >>> tess.inputs.in_file = 'aseg.mgz'
    >>> tess.inputs.label_value = 17
    >>> tess.inputs.out_file = 'lh.hippocampus'
    >>> tess.run() # doctest: +SKIP
    """
    _cmd = 'mri_tessellate'
    input_spec = MRITessellateInputSpec
    output_spec = MRITessellateOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['surface'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            return self.inputs.out_file
        else:
            _, name, ext = split_filename(self.inputs.in_file)
            return name + ext + '_' + str(self.inputs.label_value)