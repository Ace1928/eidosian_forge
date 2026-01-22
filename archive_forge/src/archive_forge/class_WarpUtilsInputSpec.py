import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpUtilsInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='--in=%s', mandatory=True, desc='Name of file containing warp-coefficients/fields. This would typically be the output from the --cout switch of fnirt (but can also use fields, like the output from --fout).')
    reference = File(exists=True, argstr='--ref=%s', mandatory=True, desc='Name of a file in target space. Note that the target space is now different from the target space that was used to create the --warp file. It would typically be the file that was specified with the --in argument when running fnirt.')
    out_format = traits.Enum('spline', 'field', argstr='--outformat=%s', desc='Specifies the output format. If set to field (default) the output will be a (4D) field-file. If set to spline the format will be a (4D) file of spline coefficients.')
    warp_resolution = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--warpres=%0.4f,%0.4f,%0.4f', desc='Specifies the resolution/knot-spacing of the splines pertaining to the coefficients in the --out file. This parameter is only relevant if --outformat is set to spline. It should be noted that if the --in file has a higher resolution, the resulting coefficients will pertain to the closest (in a least-squares sense) file in the space of fields with the --warpres resolution. It should also be noted that the resolution will always be an integer multiple of the voxel size.')
    knot_space = traits.Tuple(traits.Int, traits.Int, traits.Int, argstr='--knotspace=%d,%d,%d', desc='Alternative (to --warpres) specification of the resolution of the output spline-field.')
    out_file = File(argstr='--out=%s', position=-1, name_source=['in_file'], output_name='out_file', desc='Name of output file. The format of the output depends on what other parameters are set. The default format is a (4D) field-file. If the --outformat is set to spline the format will be a (4D) file of spline coefficients.')
    write_jacobian = traits.Bool(False, mandatory=True, usedefault=True, desc='Switch on --jac flag with automatically generated filename')
    out_jacobian = File(argstr='--jac=%s', desc='Specifies that a (3D) file of Jacobian determinants corresponding to --in should be produced and written to filename.')
    with_affine = traits.Bool(False, argstr='--withaff', desc='Specifies that the affine transform (i.e. that which was specified for the --aff parameter in fnirt) should be included as displacements in the --out file. That can be useful for interfacing with software that cannot decode FSL/fnirt coefficient-files (where the affine transform is stored separately from the displacements).')