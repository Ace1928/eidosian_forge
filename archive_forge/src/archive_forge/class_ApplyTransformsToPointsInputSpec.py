import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransformsToPointsInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='--dimensionality %d', desc='This option forces the image to be treated as a specified-dimensional image. If not specified, antsWarp tries to infer the dimensionality from the input image.')
    input_file = File(argstr='--input %s', mandatory=True, desc='Currently, the only input supported is a csv file with columns including x,y (2D), x,y,z (3D) or x,y,z,t,label (4D) column headers. The points should be defined in physical space. If in doubt how to convert coordinates from your files to the space required by antsApplyTransformsToPoints try creating/drawing a simple label volume with only one voxel set to 1 and all others set to 0. Write down the voxel coordinates. Then use ImageMaths LabelStats to find out what coordinates for this voxel antsApplyTransformsToPoints is expecting.', exists=True)
    output_file = traits.Str(argstr='--output %s', desc='Name of the output CSV file', name_source=['input_file'], hash_files=False, name_template='%s_transformed.csv')
    transforms = traits.List(File(exists=True), argstr='%s', mandatory=True, desc='transforms that will be applied to the points')
    invert_transform_flags = traits.List(traits.Bool(), desc='list indicating if a transform should be reversed')