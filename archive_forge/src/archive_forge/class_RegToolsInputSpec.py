import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegToolsInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegTools."""
    in_file = File(exists=True, desc='The input image file path', argstr='-in %s', mandatory=True)
    out_file = File(name_source=['in_file'], name_template='%s_tools.nii.gz', desc='The output file name', argstr='-out %s')
    iso_flag = traits.Bool(argstr='-iso', desc='Make output image isotropic')
    noscl_flag = traits.Bool(argstr='-noscl', desc='Set scale, slope to 0 and 1')
    mask_file = File(exists=True, desc='Values outside the mask are set to NaN', argstr='-nan %s')
    desc = 'Binarise the input image with the given threshold'
    thr_val = traits.Float(desc=desc, argstr='-thr %f')
    bin_flag = traits.Bool(argstr='-bin', desc='Binarise the input image')
    rms_val = File(exists=True, desc='Compute the mean RMS between the images', argstr='-rms %s')
    div_val = traits.Either(traits.Float, File(exists=True), desc='Divide the input by image or value', argstr='-div %s')
    mul_val = traits.Either(traits.Float, File(exists=True), desc='Multiply the input by image or value', argstr='-mul %s')
    add_val = traits.Either(traits.Float, File(exists=True), desc='Add to the input image or value', argstr='-add %s')
    sub_val = traits.Either(traits.Float, File(exists=True), desc='Add to the input image or value', argstr='-sub %s')
    down_flag = traits.Bool(desc='Downsample the image by a factor of 2', argstr='-down')
    desc = 'Smooth the input image using a cubic spline kernel'
    smo_s_val = traits.Tuple(traits.Float, traits.Float, traits.Float, desc=desc, argstr='-smoS %f %f %f')
    chg_res_val = traits.Tuple(traits.Float, traits.Float, traits.Float, desc='Change the resolution of the input image', argstr='-chgres %f %f %f')
    desc = 'Smooth the input image using a Gaussian kernel'
    smo_g_val = traits.Tuple(traits.Float, traits.Float, traits.Float, desc=desc, argstr='-smoG %f %f %f')
    inter_val = traits.Enum('NN', 'LIN', 'CUB', 'SINC', desc='Interpolation order to use to warp the floating image', argstr='-interp %d')