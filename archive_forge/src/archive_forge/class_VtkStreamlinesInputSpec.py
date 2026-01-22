import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class VtkStreamlinesInputSpec(StdOutCommandLineInputSpec):
    inputmodel = traits.Enum('raw', 'voxels', argstr='-inputmodel %s', desc='input model type (raw or voxels)', usedefault=True)
    in_file = File(exists=True, argstr=' < %s', mandatory=True, position=-2, desc='data file')
    voxeldims = traits.List(traits.Int, desc='voxel dimensions in mm', argstr='-voxeldims %s', minlen=3, maxlen=3, position=4, units='mm')
    seed_file = File(exists=False, argstr='-seedfile %s', position=1, desc='image containing seed points')
    target_file = File(exists=False, argstr='-targetfile %s', position=2, desc='image containing integer-valued target regions')
    scalar_file = File(exists=False, argstr='-scalarfile %s', position=3, desc='image that is in the same physical space as the tracts')
    colourorient = traits.Bool(argstr='-colourorient', desc='Each point on the streamline is coloured by the local orientation.')
    interpolatescalars = traits.Bool(argstr='-interpolatescalars', desc='the scalar value at each point on the streamline is calculated by trilinear interpolation')
    interpolate = traits.Bool(argstr='-interpolate', desc='the scalar value at each point on the streamline is calculated by trilinear interpolation')