import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class RESTORE(DipyDiffusionInterface):
    """
    Uses RESTORE [Chang2005]_ to perform DTI fitting with outlier detection.
    The interface uses :py:mod:`dipy`, as explained in `dipy's documentation`_.

    .. [Chang2005] Chang, LC, Jones, DK and Pierpaoli, C. RESTORE: robust     estimation of tensors by outlier rejection. MRM, 53:1088-95, (2005).

    .. _dipy's documentation:     http://nipy.org/dipy/examples_built/restore_dti.html


    Example
    -------

    >>> from nipype.interfaces import dipy as ndp
    >>> dti = ndp.RESTORE()
    >>> dti.inputs.in_file = '4d_dwi.nii'
    >>> dti.inputs.in_bval = 'bvals'
    >>> dti.inputs.in_bvec = 'bvecs'
    >>> res = dti.run() # doctest: +SKIP


    """
    input_spec = RESTOREInputSpec
    output_spec = RESTOREOutputSpec

    def _run_interface(self, runtime):
        from scipy.special import gamma
        from dipy.reconst.dti import TensorModel
        import gc
        img = nb.load(self.inputs.in_file)
        hdr = img.header.copy()
        affine = img.affine
        data = img.get_fdata()
        gtab = self._get_gradient_table()
        if isdefined(self.inputs.in_mask):
            msk = np.asanyarray(nb.load(self.inputs.in_mask).dataobj).astype(np.uint8)
        else:
            msk = np.ones(data.shape[:3], dtype=np.uint8)
        try_b0 = True
        if isdefined(self.inputs.noise_mask):
            noise_msk = nb.load(self.inputs.noise_mask).get_fdata(dtype=np.float32).reshape(-1)
            noise_msk[noise_msk > 0.5] = 1
            noise_msk[noise_msk < 1.0] = 0
            noise_msk = noise_msk.astype(np.uint8)
            try_b0 = False
        elif np.all(data[msk == 0, 0] == 0):
            IFLOGGER.info('Input data are masked.')
            noise_msk = msk.reshape(-1).astype(np.uint8)
        else:
            noise_msk = (1 - msk).reshape(-1).astype(np.uint8)
        nb0 = np.sum(gtab.b0s_mask)
        dsample = data.reshape(-1, data.shape[-1])
        if try_b0 and nb0 > 1:
            noise_data = dsample.take(np.where(gtab.b0s_mask), axis=-1)[noise_msk == 0, ...]
            n = nb0
        else:
            nodiff = np.where(~gtab.b0s_mask)
            nodiffidx = nodiff[0].tolist()
            n = 20 if len(nodiffidx) >= 20 else len(nodiffidx)
            idxs = np.random.choice(nodiffidx, size=n, replace=False)
            noise_data = dsample.take(idxs, axis=-1)[noise_msk == 1, ...]
        mean_std = np.median(noise_data.std(-1))
        try:
            bias = 1.0 - np.sqrt(2.0 / (n - 1)) * (gamma(n / 2.0) / gamma((n - 1) / 2.0))
        except:
            bias = 0.0
            pass
        sigma = mean_std * (1 + bias)
        if sigma == 0:
            IFLOGGER.warning('Noise std is 0.0, looks like data was masked and noise cannot be estimated correctly. Using default tensor model instead of RESTORE.')
            dti = TensorModel(gtab)
        else:
            IFLOGGER.info('Performing RESTORE with noise std=%.4f.', sigma)
            dti = TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
        try:
            fit_restore = dti.fit(data, msk)
        except TypeError:
            dti = TensorModel(gtab)
            fit_restore = dti.fit(data, msk)
        hdr.set_data_dtype(np.float32)
        hdr['data_type'] = 16
        for k in self._outputs().get():
            scalar = getattr(fit_restore, k)
            hdr.set_data_shape(np.shape(scalar))
            nb.Nifti1Image(scalar.astype(np.float32), affine, hdr).to_filename(self._gen_filename(k))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        for k in list(outputs.keys()):
            outputs[k] = self._gen_filename(k)
        return outputs