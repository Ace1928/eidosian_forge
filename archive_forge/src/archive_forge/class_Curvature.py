import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Curvature(FSCommand):
    """
    This program will compute the second fundamental form of a cortical
    surface. It will create two new files <hemi>.<surface>.H and
    <hemi>.<surface>.K with the mean and Gaussian curvature respectively.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import Curvature
    >>> curv = Curvature()
    >>> curv.inputs.in_file = 'lh.pial'
    >>> curv.inputs.save = True
    >>> curv.cmdline
    'mris_curvature -w lh.pial'
    """
    _cmd = 'mris_curvature'
    input_spec = CurvatureInputSpec
    output_spec = CurvatureOutputSpec

    def _format_arg(self, name, spec, value):
        if self.inputs.copy_input:
            if name == 'in_file':
                basename = os.path.basename(value)
                return spec.argstr % basename
        return super(Curvature, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.inputs.copy_input:
            in_file = os.path.basename(self.inputs.in_file)
        else:
            in_file = self.inputs.in_file
        outputs['out_mean'] = os.path.abspath(in_file) + '.H'
        outputs['out_gauss'] = os.path.abspath(in_file) + '.K'
        return outputs