from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class RescaleInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Skull-stripped image to rescale')
    ref_file = File(exists=True, mandatory=True, desc='Skull-stripped reference image')
    invert = traits.Bool(desc='Invert contrast of rescaled image')
    percentile = traits.Range(low=0.0, high=50.0, value=0.0, usedefault=True, desc='Percentile to use for reference to allow for outliers - 1 indicates the 1st and 99th percentiles in the input file will be mapped to the 99th and 1st percentiles in the reference; 0 indicates minima and maxima will be mapped')