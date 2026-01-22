import numpy as np
import nibabel as nb
from ... import logging
from ..base import TraitedSpec, File, isdefined
from .base import DipyDiffusionInterface, DipyBaseInterfaceInputSpec
class DTI(DipyDiffusionInterface):
    """
    Calculates the diffusion tensor model parameters

    Example
    -------

    >>> import nipype.interfaces.dipy as dipy
    >>> dti = dipy.DTI()
    >>> dti.inputs.in_file = 'diffusion.nii'
    >>> dti.inputs.in_bvec = 'bvecs'
    >>> dti.inputs.in_bval = 'bvals'
    >>> dti.run()                                   # doctest: +SKIP
    """
    input_spec = DTIInputSpec
    output_spec = DTIOutputSpec

    def _run_interface(self, runtime):
        from dipy.reconst import dti
        from dipy.io.utils import nifti1_symmat
        gtab = self._get_gradient_table()
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()
        affine = img.affine
        mask = None
        if isdefined(self.inputs.mask_file):
            mask = np.asanyarray(nb.load(self.inputs.mask_file).dataobj)
        tenmodel = dti.TensorModel(gtab)
        ten_fit = tenmodel.fit(data, mask)
        lower_triangular = ten_fit.lower_triangular()
        img = nifti1_symmat(lower_triangular, affine)
        out_file = self._gen_filename('dti')
        nb.save(img, out_file)
        IFLOGGER.info('DTI parameters image saved as %s', out_file)
        for metric in ['fa', 'md', 'rd', 'ad', 'color_fa']:
            data = getattr(ten_fit, metric).astype('float32')
            out_name = self._gen_filename(metric)
            nb.Nifti1Image(data, affine).to_filename(out_name)
            IFLOGGER.info('DTI %s image saved as %s', metric, out_name)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('dti')
        for metric in ['fa', 'md', 'rd', 'ad', 'color_fa']:
            outputs['{}_file'.format(metric)] = self._gen_filename(metric)
        return outputs