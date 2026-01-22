import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class N4BiasFieldCorrectionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, 4, argstr='-d %d', usedefault=True, desc='image dimension (2, 3 or 4)')
    input_image = File(argstr='--input-image %s', mandatory=True, desc='input for bias correction. Negative values or values close to zero should be processed prior to correction')
    mask_image = File(argstr='--mask-image %s', desc='image to specify region to perform final bias correction in')
    weight_image = File(argstr='--weight-image %s', desc='image for relative weighting (e.g. probability map of the white matter) of voxels during the B-spline fitting. ')
    output_image = traits.Str(argstr='--output %s', desc='output file name', name_source=['input_image'], name_template='%s_corrected', keep_extension=True, hash_files=False)
    bspline_fitting_distance = traits.Float(argstr='--bspline-fitting %s')
    bspline_order = traits.Int(requires=['bspline_fitting_distance'])
    shrink_factor = traits.Int(argstr='--shrink-factor %d')
    n_iterations = traits.List(traits.Int(), argstr='--convergence %s')
    convergence_threshold = traits.Float(requires=['n_iterations'])
    save_bias = traits.Bool(False, mandatory=True, usedefault=True, desc='True if the estimated bias should be saved to file.', xor=['bias_image'])
    bias_image = File(desc='Filename for the estimated bias.', hash_files=False)
    copy_header = traits.Bool(False, mandatory=True, usedefault=True, desc='copy headers of the original image into the output (corrected) file')
    rescale_intensities = traits.Bool(False, usedefault=True, argstr='-r', min_ver='2.1.0', desc='[NOTE: Only ANTs>=2.1.0]\nAt each iteration, a new intensity mapping is calculated and applied but there\nis nothing which constrains the new intensity range to be within certain values.\nThe result is that the range can "drift" from the original at each iteration.\nThis option rescales to the [min,max] range of the original image intensities\nwithin the user-specified mask.')
    histogram_sharpening = traits.Tuple((0.15, 0.01, 200), traits.Float, traits.Float, traits.Int, argstr='--histogram-sharpening [%g,%g,%d]', desc='Three-values tuple of histogram sharpening parameters (FWHM, wienerNose, numberOfHistogramBins).\nThese options describe the histogram sharpening parameters, i.e. the deconvolution step parameters described in the original N3 algorithm.\nThe default values have been shown to work fairly well.')