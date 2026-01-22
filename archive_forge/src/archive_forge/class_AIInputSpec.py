import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AIInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, usedefault=True, argstr='-d %d', desc='dimension of output image')
    verbose = traits.Bool(False, usedefault=True, argstr='-v %d', desc='enable verbosity')
    fixed_image = File(exists=True, mandatory=True, desc='Image to which the moving_image should be transformed')
    moving_image = File(exists=True, mandatory=True, desc='Image that will be transformed to fixed_image')
    fixed_image_mask = File(exists=True, argstr='-x %s', desc='fixed mage mask')
    moving_image_mask = File(exists=True, requires=['fixed_image_mask'], desc='moving mage mask')
    metric_trait = (traits.Enum('Mattes', 'GC', 'MI'), traits.Int(32), traits.Enum('Regular', 'Random', 'None'), traits.Range(value=0.2, low=0.0, high=1.0))
    metric = traits.Tuple(*metric_trait, argstr='-m %s', mandatory=True, desc='the metric(s) to use.')
    transform = traits.Tuple(traits.Enum('Affine', 'Rigid', 'Similarity'), traits.Range(value=0.1, low=0.0, exclude_low=True), argstr='-t %s[%g]', usedefault=True, desc='Several transform options are available')
    principal_axes = traits.Bool(False, usedefault=True, argstr='-p %d', xor=['blobs'], desc='align using principal axes')
    search_factor = traits.Tuple(traits.Float(20), traits.Range(value=0.12, low=0.0, high=1.0), usedefault=True, argstr='-s [%g,%g]', desc='search factor')
    search_grid = traits.Either(traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float, traits.Float)), traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float)), argstr='-g %s', desc='Translation search grid in mm', min_ver='2.3.0')
    convergence = traits.Tuple(traits.Range(low=1, high=10000, value=10), traits.Float(1e-06), traits.Range(low=1, high=100, value=10), usedefault=True, argstr='-c [%d,%g,%d]', desc='convergence')
    output_transform = File('initialization.mat', usedefault=True, argstr='-o %s', desc='output file name')