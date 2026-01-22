import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTITrackerInputSpec(CommandLineInputSpec):
    tensor_file = File(exists=True, desc='reconstructed tensor file')
    input_type = traits.Enum('nii', 'analyze', 'ni1', 'nii.gz', desc="Input and output file type. Accepted values are:\n\n* analyze -> analyze format 7.5\n* ni1     -> nifti format saved in separate .hdr and .img file\n* nii     -> nifti format with one .nii file\n* nii.gz  -> nifti format with compression\n\nDefault type is 'nii'\n", argstr='-it %s')
    tracking_method = traits.Enum('fact', 'rk2', 'tl', 'sl', desc='Tracking algorithm.\n\n* fact -> use FACT method for tracking. This is the default method.\n* rk2  -> use 2nd order Runge-Kutta method for tracking.\n* tl   -> use tensorline method for tracking.\n* sl   -> use interpolated streamline method with fixed step-length\n\n', argstr='-%s')
    step_length = traits.Float(desc='Step length, in the unit of minimum voxel size.\ndefault value is 0.5 for interpolated streamline method\nand 0.1 for other methods', argstr='-l %f')
    angle_threshold = traits.Float(desc='set angle threshold. default value is 35 degree', argstr='-at %f')
    angle_threshold_weight = traits.Float(desc='set angle threshold weighting factor. weighting will be applied on top of the angle_threshold', argstr='-atw %f')
    random_seed = traits.Int(desc='use random location in a voxel instead of the center of the voxel to seed. can also define number of seed per voxel. default is 1', argstr='-rseed %d')
    invert_x = traits.Bool(desc='invert x component of the vector', argstr='-ix')
    invert_y = traits.Bool(desc='invert y component of the vector', argstr='-iy')
    invert_z = traits.Bool(desc='invert z component of the vector', argstr='-iz')
    swap_xy = traits.Bool(desc='swap x & y vectors while tracking', argstr='-sxy')
    swap_yz = traits.Bool(desc='swap y & z vectors while tracking', argstr='-syz')
    swap_zx = traits.Bool(desc='swap x & z vectors while tracking', argstr='-szx')
    mask1_file = File(desc='first mask image', mandatory=True, argstr='-m %s', position=2)
    mask1_threshold = traits.Float(desc='threshold value for the first mask image, if not given, the program will try automatically find the threshold', position=3)
    mask2_file = File(desc='second mask image', argstr='-m2 %s', position=4)
    mask2_threshold = traits.Float(desc='threshold value for the second mask image, if not given, the program will try automatically find the threshold', position=5)
    input_data_prefix = traits.Str('dti', desc='for internal naming use only', position=0, argstr='%s', usedefault=True)
    output_file = File('tracks.trk', 'file containing tracks', argstr='%s', position=1, usedefault=True)
    output_mask = File(desc='output a binary mask file in analyze format', argstr='-om %s')
    primary_vector = traits.Enum('v2', 'v3', desc='which vector to use for fibre tracking: v2 or v3. If not set use v1', argstr='-%s')