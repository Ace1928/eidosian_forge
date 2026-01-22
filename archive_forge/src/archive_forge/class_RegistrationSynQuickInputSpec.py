import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class RegistrationSynQuickInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=True, desc='image dimension (2 or 3)')
    fixed_image = InputMultiPath(File(exists=True), mandatory=True, argstr='-f %s...', desc='Fixed image or source image or reference image')
    moving_image = InputMultiPath(File(exists=True), mandatory=True, argstr='-m %s...', desc='Moving image or target image')
    output_prefix = Str('transform', usedefault=True, argstr='-o %s', desc='A prefix that is prepended to all output files')
    num_threads = traits.Int(default_value=LOCAL_DEFAULT_NUMBER_OF_THREADS, usedefault=True, desc='Number of threads (default = 1)', argstr='-n %d')
    transform_type = traits.Enum('s', 't', 'r', 'a', 'sr', 'b', 'br', argstr='-t %s', desc='Transform type\n\n  * t:  translation\n  * r:  rigid\n  * a:  rigid + affine\n  * s:  rigid + affine + deformable syn (default)\n  * sr: rigid + deformable syn\n  * b:  rigid + affine + deformable b-spline syn\n  * br: rigid + deformable b-spline syn\n\n', usedefault=True)
    use_histogram_matching = traits.Bool(False, argstr='-j %d', desc='use histogram matching')
    histogram_bins = traits.Int(default_value=32, usedefault=True, argstr='-r %d', desc='histogram bins for mutual information in SyN stage                                  (default = 32)')
    spline_distance = traits.Int(default_value=26, usedefault=True, argstr='-s %d', desc='spline distance for deformable B-spline SyN transform                                  (default = 26)')
    precision_type = traits.Enum('double', 'float', argstr='-p %s', desc='precision type (default = double)', usedefault=True)
    random_seed = traits.Int(argstr='-e %d', desc='fixed random seed', min_ver='2.3.0')