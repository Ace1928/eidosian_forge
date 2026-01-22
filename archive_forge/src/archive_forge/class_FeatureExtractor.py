from ..base import (
import os
class FeatureExtractor(CommandLine):
    """
    Extract features (for later training and/or classifying)
    """
    input_spec = FeatureExtractorInputSpec
    output_spec = FeatureExtractorOutputSpec
    cmd = 'fix -f'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['mel_ica'] = self.inputs.mel_ica
        return outputs