import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AverageImagesInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', mandatory=True, position=0, desc='image dimension (2 or 3)')
    output_average_image = File('average.nii', argstr='%s', position=1, usedefault=True, hash_files=False, desc='the name of the resulting image.')
    normalize = traits.Bool(argstr='%d', mandatory=True, position=2, desc='Normalize: if true, the 2nd image is divided by its mean. This will select the largest image to average into.')
    images = InputMultiObject(File(exists=True), argstr='%s', mandatory=True, position=3, desc='image to apply transformation to (generally a coregistered functional)')