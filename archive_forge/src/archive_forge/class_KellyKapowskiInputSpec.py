import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class KellyKapowskiInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='--image-dimensionality %d', usedefault=True, desc='image dimension (2 or 3)')
    segmentation_image = File(exists=True, argstr='--segmentation-image "%s"', mandatory=True, desc='A segmentation image must be supplied labeling the gray and white matters. Default values = 2 and 3, respectively.')
    gray_matter_label = traits.Int(2, usedefault=True, desc='The label value for the gray matter label in the segmentation_image.')
    white_matter_label = traits.Int(3, usedefault=True, desc='The label value for the white matter label in the segmentation_image.')
    gray_matter_prob_image = File(exists=True, argstr='--gray-matter-probability-image "%s"', desc='In addition to the segmentation image, a gray matter probability image can be used. If no such image is supplied, one is created using the segmentation image and a variance of 1.0 mm.')
    white_matter_prob_image = File(exists=True, argstr='--white-matter-probability-image "%s"', desc='In addition to the segmentation image, a white matter probability image can be used. If no such image is supplied, one is created using the segmentation image and a variance of 1.0 mm.')
    convergence = traits.Str('[50,0.001,10]', argstr='--convergence "%s"', usedefault=True, desc='Convergence is determined by fitting a line to the normalized energy profile of the last N iterations (where N is specified by the window size) and determining the slope which is then compared with the convergence threshold.')
    thickness_prior_estimate = traits.Float(10, usedefault=True, argstr='--thickness-prior-estimate %f', desc='Provides a prior constraint on the final thickness measurement in mm.')
    thickness_prior_image = File(exists=True, argstr='--thickness-prior-image "%s"', desc='An image containing spatially varying prior thickness values.')
    gradient_step = traits.Float(0.025, usedefault=True, argstr='--gradient-step %f', desc='Gradient step size for the optimization.')
    smoothing_variance = traits.Float(1.0, usedefault=True, argstr='--smoothing-variance %f', desc='Defines the Gaussian smoothing of the hit and total images.')
    smoothing_velocity_field = traits.Float(1.5, usedefault=True, argstr='--smoothing-velocity-field-parameter %f', desc='Defines the Gaussian smoothing of the velocity field (default = 1.5). If the b-spline smoothing option is chosen, then this defines the isotropic mesh spacing for the smoothing spline (default = 15).')
    use_bspline_smoothing = traits.Bool(argstr='--use-bspline-smoothing 1', desc='Sets the option for B-spline smoothing of the velocity field.')
    number_integration_points = traits.Int(10, usedefault=True, argstr='--number-of-integration-points %d', desc='Number of compositions of the diffeomorphism per iteration.')
    max_invert_displacement_field_iters = traits.Int(20, usedefault=True, argstr='--maximum-number-of-invert-displacement-field-iterations %d', desc='Maximum number of iterations for estimating the invertdisplacement field.')
    cortical_thickness = File(argstr='--output "%s"', keep_extension=True, name_source=['segmentation_image'], name_template='%s_cortical_thickness', desc='Filename for the cortical thickness.', hash_files=False)
    warped_white_matter = File(name_source=['segmentation_image'], keep_extension=True, name_template='%s_warped_white_matter', desc='Filename for the warped white matter file.', hash_files=False)