import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class PointsWarp(CommandLine):
    """Use ``transformix`` to apply a transform on an input point set.
    The transform is specified in the transform-parameter file.

    Example
    -------

    >>> from nipype.interfaces.elastix import PointsWarp
    >>> reg = PointsWarp()
    >>> reg.inputs.points_file = 'surf1.vtk'
    >>> reg.inputs.transform_file = 'TransformParameters.0.txt'
    >>> reg.cmdline
    'transformix -threads 1 -out ./ -def surf1.vtk -tp TransformParameters.0.txt'


    """
    _cmd = 'transformix'
    input_spec = PointsWarpInputSpec
    output_spec = PointsWarpOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = op.abspath(self.inputs.output_path)
        fname, ext = op.splitext(op.basename(self.inputs.points_file))
        outputs['warped_file'] = op.join(out_dir, 'outputpoints%s' % ext)
        return outputs