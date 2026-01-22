import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SmoothTessellation(FSCommand):
    """
    Smooth a tessellated surface.

    See Also
    --------
    `nipype.interfaces.freesurfer.utils.SurfaceSmooth`_ interface for smoothing a scalar field
    along a surface manifold

    Example
    -------
    >>> import nipype.interfaces.freesurfer as fs
    >>> smooth = fs.SmoothTessellation()
    >>> smooth.inputs.in_file = 'lh.hippocampus.stl'
    >>> smooth.run() # doctest: +SKIP

    """
    _cmd = 'mris_smooth'
    input_spec = SmoothTessellationInputSpec
    output_spec = SmoothTessellationOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['surface'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            return os.path.abspath(self.inputs.out_file)
        else:
            _, name, ext = split_filename(self.inputs.in_file)
            return os.path.abspath(name + '_smoothed' + ext)

    def _run_interface(self, runtime):
        runtime = super(SmoothTessellation, self)._run_interface(runtime)
        if 'failed' in runtime.stderr:
            self.raise_exception(runtime)
        return runtime