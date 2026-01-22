import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ConvertWarpInputSpec(FSLCommandInputSpec):
    reference = File(exists=True, argstr='--ref=%s', mandatory=True, position=1, desc='Name of a file in target space of the full transform.')
    out_file = File(argstr='--out=%s', position=-1, name_source=['reference'], name_template='%s_concatwarp', output_name='out_file', desc='Name of output file, containing warps that are the combination of all those given as arguments. The format of this will be a field-file (rather than spline coefficients) with any affine components included.')
    premat = File(exists=True, argstr='--premat=%s', desc='filename for pre-transform (affine matrix)')
    warp1 = File(exists=True, argstr='--warp1=%s', desc='Name of file containing initial warp-fields/coefficients (follows premat). This could e.g. be a fnirt-transform from a subjects structural scan to an average of a group of subjects.')
    midmat = File(exists=True, argstr='--midmat=%s', desc='Name of file containing mid-warp-affine transform')
    warp2 = File(exists=True, argstr='--warp2=%s', desc='Name of file containing secondary warp-fields/coefficients (after warp1/midmat but before postmat). This could e.g. be a fnirt-transform from the average of a group of subjects to some standard space (e.g. MNI152).')
    postmat = File(exists=True, argstr='--postmat=%s', desc='Name of file containing an affine transform (applied last). It could e.g. be an affine transform that maps the MNI152-space into a better approximation to the Talairach-space (if indeed there is one).')
    shift_in_file = File(exists=True, argstr='--shiftmap=%s', desc='Name of file containing a "shiftmap", a non-linear transform with displacements only in one direction (applied first, before premat). This would typically be a fieldmap that has been pre-processed using fugue that maps a subjects functional (EPI) data onto an undistorted space (i.e. a space that corresponds to his/her true anatomy).')
    shift_direction = traits.Enum('y-', 'y', 'x', 'x-', 'z', 'z-', argstr='--shiftdir=%s', requires=['shift_in_file'], desc='Indicates the direction that the distortions from --shiftmap goes. It depends on the direction and polarity of the phase-encoding in the EPI sequence.')
    cons_jacobian = traits.Bool(False, argstr='--constrainj', desc='Constrain the Jacobian of the warpfield to lie within specified min/max limits.')
    jacobian_min = traits.Float(argstr='--jmin=%f', desc='Minimum acceptable Jacobian value for constraint (default 0.01)')
    jacobian_max = traits.Float(argstr='--jmax=%f', desc='Maximum acceptable Jacobian value for constraint (default 100.0)')
    abswarp = traits.Bool(argstr='--abs', xor=['relwarp'], desc='If set it indicates that the warps in --warp1 and --warp2 should be interpreted as absolute. I.e. the values in --warp1/2 are the coordinates in the next space, rather than displacements. This flag is ignored if --warp1/2 was created by fnirt, which always creates relative displacements.')
    relwarp = traits.Bool(argstr='--rel', xor=['abswarp'], desc='If set it indicates that the warps in --warp1/2 should be interpreted as relative. I.e. the values in --warp1/2 are displacements from the coordinates in the next space.')
    out_abswarp = traits.Bool(argstr='--absout', xor=['out_relwarp'], desc='If set it indicates that the warps in --out should be absolute, i.e. the values in --out are displacements from the coordinates in --ref.')
    out_relwarp = traits.Bool(argstr='--relout', xor=['out_abswarp'], desc='If set it indicates that the warps in --out should be relative, i.e. the values in --out are displacements from the coordinates in --ref.')