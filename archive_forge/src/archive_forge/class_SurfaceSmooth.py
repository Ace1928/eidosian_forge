import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSmooth(FSCommand):
    """Smooth a surface image with mri_surf2surf.

    The surface is smoothed by an iterative process of averaging the
    value at each vertex with those of its adjacent neighbors. You may supply
    either the number of iterations to run or a desired effective FWHM of the
    smoothing process.  If the latter, the underlying program will calculate
    the correct number of iterations internally.

    See Also
    --------
    `nipype.interfaces.freesurfer.utils.SmoothTessellation`_ interface for
    smoothing a tessellated surface (e.g. in gifti or .stl)

    Examples
    --------
    >>> import nipype.interfaces.freesurfer as fs
    >>> smoother = fs.SurfaceSmooth()
    >>> smoother.inputs.in_file = "lh.cope1.mgz"
    >>> smoother.inputs.subject_id = "subj_1"
    >>> smoother.inputs.hemi = "lh"
    >>> smoother.inputs.fwhm = 5
    >>> smoother.cmdline # doctest: +ELLIPSIS
    'mri_surf2surf --cortex --fwhm 5.0000 --hemi lh --sval lh.cope1.mgz --tval ...lh.cope1_smooth5.mgz --s subj_1'
    >>> smoother.run() # doctest: +SKIP

    """
    _cmd = 'mri_surf2surf'
    input_spec = SurfaceSmoothInputSpec
    output_spec = SurfaceSmoothOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            in_file = self.inputs.in_file
            if isdefined(self.inputs.fwhm):
                kernel = self.inputs.fwhm
            else:
                kernel = self.inputs.smooth_iters
            outputs['out_file'] = fname_presuffix(in_file, suffix='_smooth%d' % kernel, newpath=os.getcwd())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None