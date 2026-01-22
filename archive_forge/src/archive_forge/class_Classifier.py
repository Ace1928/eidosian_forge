from ..base import (
import os
class Classifier(CommandLine):
    """
    Classify ICA components using a specific training dataset (<thresh> is in the range 0-100, typically 5-20).
    """
    input_spec = ClassifierInputSpec
    output_spec = ClassifierOutputSpec
    cmd = 'fix -c'

    def _gen_artifacts_list_file(self, mel_ica, thresh):
        _, trained_wts_file = os.path.split(self.inputs.trained_wts_file)
        trained_wts_filestem = trained_wts_file.split('.')[0]
        filestem = 'fix4melview_' + trained_wts_filestem + '_thr'
        fname = os.path.join(mel_ica, filestem + str(thresh) + '.txt')
        return fname

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['artifacts_list_file'] = self._gen_artifacts_list_file(self.inputs.mel_ica, self.inputs.thresh)
        return outputs