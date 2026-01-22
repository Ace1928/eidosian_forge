import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class LabelConfigInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input anatomical image')
    in_config = File(exists=True, argstr='%s', position=-2, desc='connectome configuration file')
    out_file = File('parcellation.mif', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output file after processing')
    lut_basic = File(argstr='-lut_basic %s', desc='get information from a basic lookup table consisting of index / name pairs')
    lut_fs = File(argstr='-lut_freesurfer %s', desc='get information from a FreeSurfer lookup table(typically "FreeSurferColorLUT.txt")')
    lut_aal = File(argstr='-lut_aal %s', desc='get information from the AAL lookup table (typically "ROI_MNI_V4.txt")')
    lut_itksnap = File(argstr='-lut_itksnap %s', desc='get information from an ITK - SNAP lookup table(this includes the IIT atlas file "LUT_GM.txt")')
    spine = File(argstr='-spine %s', desc='provide a manually-defined segmentation of the base of the spine where the streamlines terminate, so that this can become a node in the connection matrix.')
    nthreads = traits.Int(argstr='-nthreads %d', desc='number of threads. if zero, the number of available cpus will be used', nohash=True)