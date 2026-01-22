import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class ConvertScalarImageToRGBInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True, desc='image dimension (2 or 3)', mandatory=True, position=0)
    input_image = File(argstr='%s', exists=True, desc='Main input is a 3-D grayscale image.', mandatory=True, position=1)
    output_image = traits.Str('rgb.nii.gz', argstr='%s', usedefault=True, desc='rgb output image', position=2)
    mask_image = traits.Either('none', traits.File(exists=True), argstr='%s', desc='mask image', position=3, default='none', usedefault=True)
    colormap = traits.Enum('grey', 'red', 'green', 'blue', 'copper', 'jet', 'hsv', 'spring', 'summer', 'autumn', 'winter', 'hot', 'cool', 'overunder', 'custom', argstr='%s', desc='Select a colormap', mandatory=True, position=4)
    custom_color_map_file = traits.Str('none', argstr='%s', usedefault=True, desc='custom color map file', position=5)
    minimum_input = traits.Int(argstr='%d', desc='minimum input', mandatory=True, position=6)
    maximum_input = traits.Int(argstr='%d', desc='maximum input', mandatory=True, position=7)
    minimum_RGB_output = traits.Int(0, usedefault=True, argstr='%d', position=8)
    maximum_RGB_output = traits.Int(255, usedefault=True, argstr='%d', position=9)