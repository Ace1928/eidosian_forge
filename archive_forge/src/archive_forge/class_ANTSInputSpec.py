import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class ANTSInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', position=1, desc='image dimension (2 or 3)')
    fixed_image = InputMultiPath(File(exists=True), mandatory=True, desc='image to which the moving image is warped')
    moving_image = InputMultiPath(File(exists=True), argstr='%s', mandatory=True, desc='image to apply transformation to (generally a coregisteredfunctional)')
    metric = traits.List(traits.Enum('CC', 'MI', 'SMI', 'PR', 'SSD', 'MSQ', 'PSE'), mandatory=True, desc='')
    metric_weight = traits.List(traits.Float(), value=[1.0], usedefault=True, requires=['metric'], mandatory=True, desc='the metric weight(s) for each stage. The weights must sum to 1 per stage.')
    radius = traits.List(traits.Int(), requires=['metric'], mandatory=True, desc='radius of the region (i.e. number of layers around a voxel/pixel) that is used for computing cross correlation')
    output_transform_prefix = Str('out', usedefault=True, argstr='--output-naming %s', mandatory=True, desc='')
    transformation_model = traits.Enum('Diff', 'Elast', 'Exp', 'Greedy Exp', 'SyN', argstr='%s', mandatory=True, desc='')
    gradient_step_length = traits.Float(requires=['transformation_model'], desc='')
    number_of_time_steps = traits.Int(requires=['gradient_step_length'], desc='')
    delta_time = traits.Float(requires=['number_of_time_steps'], desc='')
    symmetry_type = traits.Float(requires=['delta_time'], desc='')
    use_histogram_matching = traits.Bool(argstr='%s', default_value=True, usedefault=True)
    number_of_iterations = traits.List(traits.Int(), argstr='--number-of-iterations %s', sep='x')
    smoothing_sigmas = traits.List(traits.Int(), argstr='--gaussian-smoothing-sigmas %s', sep='x')
    subsampling_factors = traits.List(traits.Int(), argstr='--subsampling-factors %s', sep='x')
    affine_gradient_descent_option = traits.List(traits.Float(), argstr='%s')
    mi_option = traits.List(traits.Int(), argstr='--MI-option %s', sep='x')
    regularization = traits.Enum('Gauss', 'DMFFD', argstr='%s', desc='')
    regularization_gradient_field_sigma = traits.Float(requires=['regularization'], desc='')
    regularization_deformation_field_sigma = traits.Float(requires=['regularization'], desc='')
    number_of_affine_iterations = traits.List(traits.Int(), argstr='--number-of-affine-iterations %s', sep='x')