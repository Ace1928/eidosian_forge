import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class SplitNifti(NiftiGeneratorBase):
    """
    Split one Nifti file into many along the specified dimension. Each
    result has an updated meta data extension as well.
    """
    input_spec = SplitNiftiInputSpec
    output_spec = SplitNiftiOutputSpec

    def _run_interface(self, runtime):
        self.out_list = []
        nii = nb.load(self.inputs.in_file)
        nw = NiftiWrapper(nii, make_empty=True)
        split_dim = None
        if self.inputs.split_dim == traits.Undefined:
            split_dim = None
        else:
            split_dim = self.inputs.split_dim
        for split_idx, split_nw in enumerate(nw.split(split_dim)):
            const_meta = split_nw.meta_ext.get_class_dict(('global', 'const'))
            out_path = self._get_out_path(const_meta, idx=split_idx)
            nb.save(split_nw.nii_img, out_path)
            self.out_list.append(out_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_list'] = self.out_list
        return outputs