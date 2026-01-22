import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class MeasureImageSimilarityInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='--dimensionality %d', position=1, desc='Dimensionality of the fixed/moving image pair')
    fixed_image = File(exists=True, mandatory=True, desc='Image to which the moving image is warped')
    moving_image = File(exists=True, mandatory=True, desc='Image to apply transformation to (generally a coregistered functional)')
    metric = traits.Enum('CC', 'MI', 'Mattes', 'MeanSquares', 'Demons', 'GC', argstr='%s', mandatory=True)
    metric_weight = traits.Float(requires=['metric'], default_value=1.0, usedefault=True, desc='The "metricWeight" variable is not used.')
    radius_or_number_of_bins = traits.Int(requires=['metric'], mandatory=True, desc='The number of bins in each stage for the MI and Mattes metric, or the radius for other metrics')
    sampling_strategy = traits.Enum('None', 'Regular', 'Random', requires=['metric'], usedefault=True, desc='Manner of choosing point set over which to optimize the metric. Defaults to "None" (i.e. a dense sampling of one sample per voxel).')
    sampling_percentage = traits.Either(traits.Range(low=0.0, high=1.0), requires=['metric'], mandatory=True, desc='Percentage of points accessible to the sampling strategy over which to optimize the metric.')
    fixed_image_mask = File(exists=True, argstr='%s', desc='mask used to limit metric sampling region of the fixed image')
    moving_image_mask = File(exists=True, requires=['fixed_image_mask'], desc='mask used to limit metric sampling region of the moving image')