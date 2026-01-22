import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ImageMaths(FSLCommand):
    """Use FSL fslmaths command to allow mathematical manipulation of images
    `FSL info <http://www.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro/index.htm#fslutils>`_


    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import anatfile
    >>> maths = fsl.ImageMaths(in_file=anatfile, op_string= '-add 5',
    ...                        out_file='foo_maths.nii')
    >>> maths.cmdline == 'fslmaths %s -add 5 foo_maths.nii' % anatfile
    True


    """
    input_spec = ImageMathsInputSpec
    output_spec = ImageMathsOutputSpec
    _cmd = 'fslmaths'

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _parse_inputs(self, skip=None):
        return super(ImageMaths, self)._parse_inputs(skip=['suffix'])

    def _list_outputs(self):
        suffix = '_maths'
        if isdefined(self.inputs.suffix):
            suffix = self.inputs.suffix
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix=suffix)
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs