import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class PointsWarpInputSpec(ElastixBaseInputSpec):
    points_file = File(exists=True, argstr='-def %s', mandatory=True, desc='input points (accepts .vtk triangular meshes).')
    transform_file = File(exists=True, mandatory=True, argstr='-tp %s', desc='transform-parameter file, only 1')