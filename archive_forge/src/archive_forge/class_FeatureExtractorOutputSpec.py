from ..base import (
import os
class FeatureExtractorOutputSpec(TraitedSpec):
    mel_ica = Directory(exists=True, copyfile=False, desc='Melodic output directory or directories', argstr='%s', position=-1)