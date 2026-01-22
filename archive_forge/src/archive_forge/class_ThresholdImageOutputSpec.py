import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ThresholdImageOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='resampled file')