import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class WarpImageMultiTransformInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True, desc='image dimension (2 or 3)', position=1)
    input_image = File(argstr='%s', mandatory=True, desc='image to apply transformation to (generally a coregistered functional)', position=2)
    output_image = File(genfile=True, hash_files=False, argstr='%s', desc='name of the output warped image', position=3, xor=['out_postfix'])
    out_postfix = File('_wimt', usedefault=True, hash_files=False, desc='Postfix that is prepended to all output files (default = _wimt)', xor=['output_image'])
    reference_image = File(argstr='-R %s', xor=['tightest_box'], desc='reference image space that you wish to warp INTO')
    tightest_box = traits.Bool(argstr='--tightest-bounding-box', desc='computes tightest bounding box (overridden by reference_image if given)', xor=['reference_image'])
    reslice_by_header = traits.Bool(argstr='--reslice-by-header', desc='Uses orientation matrix and origin encoded in reference image file header. Not typically used with additional transforms')
    use_nearest = traits.Bool(argstr='--use-NN', desc='Use nearest neighbor interpolation')
    use_bspline = traits.Bool(argstr='--use-BSpline', desc='Use 3rd order B-Spline interpolation')
    transformation_series = InputMultiObject(File(exists=True), argstr='%s', desc='transformation file(s) to be applied', mandatory=True, position=-1)
    invert_affine = traits.List(traits.Int, desc='List of Affine transformations to invert.E.g.: [1,4,5] inverts the 1st, 4th, and 5th Affines found in transformation_series. Note that indexing starts with 1 and does not include warp fields. Affine transformations are distinguished from warp fields by the word "affine" included in their filenames.')