import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class StatsOutput(TraitedSpec):
    """Output Spec for seg_stats interfaces."""
    output = traits.Array(desc='Output array from seg_stats')