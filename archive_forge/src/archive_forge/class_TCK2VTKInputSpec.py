import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TCK2VTKInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input tractography')
    out_file = File('tracks.vtk', argstr='%s', usedefault=True, position=-1, desc='output VTK file')
    reference = File(exists=True, argstr='-image %s', desc='if specified, the properties of this image will be used to convert track point positions from real (scanner) coordinates into image coordinates (in mm).')
    voxel = File(exists=True, argstr='-image %s', desc='if specified, the properties of this image will be used to convert track point positions from real (scanner) coordinates into image coordinates.')
    nthreads = traits.Int(argstr='-nthreads %d', desc='number of threads. if zero, the number of available cpus will be used', nohash=True)