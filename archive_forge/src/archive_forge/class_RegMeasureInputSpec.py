import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegMeasureInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegMeasure."""
    ref_file = File(exists=True, desc='The input reference/target image', argstr='-ref %s', mandatory=True)
    flo_file = File(exists=True, desc='The input floating/source image', argstr='-flo %s', mandatory=True)
    measure_type = traits.Enum('ncc', 'lncc', 'nmi', 'ssd', mandatory=True, argstr='-%s', desc='Measure of similarity to compute')
    out_file = File(name_source=['flo_file'], name_template='%s', argstr='-out %s', desc='The output text file containing the measure')