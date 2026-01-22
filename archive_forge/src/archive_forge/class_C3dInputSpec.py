import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
class C3dInputSpec(CommandLineInputSpec):
    in_file = InputMultiPath(File(), position=1, argstr='%s', mandatory=True, desc='Input file (wildcard and multiple are supported).')
    out_file = File(exists=False, argstr='-o %s', position=-1, xor=['out_files'], desc='Output file of last image on the stack.')
    out_files = InputMultiPath(File(), argstr='-oo %s', xor=['out_file'], position=-1, desc='Write all images on the convert3d stack as multiple files. Supports both list of output files or a pattern for the output filenames (using %d substitution).')
    pix_type = traits.Enum('float', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'double', argstr='-type %s', desc='Specifies the pixel type for the output image. By default, images are written in floating point (float) format')
    scale = traits.Either(traits.Int(), traits.Float(), argstr='-scale %s', desc='Multiplies the intensity of each voxel in the last image on the stack by the given factor.')
    shift = traits.Either(traits.Int(), traits.Float(), argstr='-shift %s', desc='Adds the given constant to every voxel.')
    interp = traits.Enum('Linear', 'NearestNeighbor', 'Cubic', 'Sinc', 'Gaussian', argstr='-interpolation %s', desc='Specifies the interpolation used with -resample and other commands. Default is Linear.')
    resample = traits.Str(argstr='-resample %s', desc='Resamples the image, keeping the bounding box the same, but changing the number of voxels in the image. The dimensions can be specified as a percentage, for example to double the number of voxels in each direction. The -interpolation flag affects how sampling is performed.')
    smooth = traits.Str(argstr='-smooth %s', desc='Applies Gaussian smoothing to the image. The parameter vector specifies the standard deviation of the Gaussian kernel.')
    multicomp_split = traits.Bool(False, usedefault=True, argstr='-mcr', position=0, desc='Enable reading of multi-component images.')
    is_4d = traits.Bool(False, usedefault=True, desc='Changes command to support 4D file operations (default is false).')