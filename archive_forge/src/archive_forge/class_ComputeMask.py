import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class ComputeMask(NipyBaseInterface):
    input_spec = ComputeMaskInputSpec
    output_spec = ComputeMaskOutputSpec

    def _run_interface(self, runtime):
        from nipy.labs.mask import compute_mask
        args = {}
        for key in [k for k, _ in list(self.inputs.items()) if k not in BaseInterfaceInputSpec().trait_names()]:
            value = getattr(self.inputs, key)
            if isdefined(value):
                if key in ['mean_volume', 'reference_volume']:
                    value = np.asanyarray(nb.load(value).dataobj)
                args[key] = value
        brain_mask = compute_mask(**args)
        _, name, ext = split_filename(self.inputs.mean_volume)
        self._brain_mask_path = os.path.abspath('%s_mask.%s' % (name, ext))
        nb.save(nb.Nifti1Image(brain_mask.astype(np.uint8), nii.affine), self._brain_mask_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['brain_mask'] = self._brain_mask_path
        return outputs