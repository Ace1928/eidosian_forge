import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class DWI2SphericalHarmonicsImageInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='Diffusion-weighted images')
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output filename')
    encoding_file = File(exists=True, argstr='-grad %s', mandatory=True, position=1, desc='Gradient encoding, supplied as a 4xN text file with each line is in the format [ X Y Z b ], where [ X Y Z ] describe the direction of the applied gradient, and b gives the b-value in units (1000 s/mm^2). See FSL2MRTrix')
    maximum_harmonic_order = traits.Float(argstr='-lmax %s', desc='set the maximum harmonic order for the output series. By default, the program will use the highest possible lmax given the number of diffusion-weighted images.')
    normalise = traits.Bool(argstr='-normalise', position=3, desc='normalise the DW signal to the b=0 image')