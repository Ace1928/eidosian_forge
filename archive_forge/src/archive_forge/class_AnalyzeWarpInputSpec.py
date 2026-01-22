import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class AnalyzeWarpInputSpec(ApplyWarpInputSpec):
    points = traits.Enum('all', usedefault=True, position=0, argstr='-def %s', desc='transform all points from the input-image, which effectively generates a deformation field.')
    jac = traits.Enum('all', usedefault=True, argstr='-jac %s', desc='generate an image with the determinant of the spatial Jacobian')
    jacmat = traits.Enum('all', usedefault=True, argstr='-jacmat %s', desc='generate an image with the spatial Jacobian matrix at each voxel')
    moving_image = File(exists=True, argstr='-in %s', desc='input image to deform (not used)')