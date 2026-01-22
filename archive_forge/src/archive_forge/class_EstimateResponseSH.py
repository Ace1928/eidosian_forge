import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class EstimateResponseSH(DipyDiffusionInterface):
    """
    Uses dipy to compute the single fiber response to be used in spherical
    deconvolution methods, in a similar way to MRTrix's command
    ``estimate_response``.


    Example
    -------

    >>> from nipype.interfaces import dipy as ndp
    >>> dti = ndp.EstimateResponseSH()
    >>> dti.inputs.in_file = '4d_dwi.nii'
    >>> dti.inputs.in_bval = 'bvals'
    >>> dti.inputs.in_bvec = 'bvecs'
    >>> dti.inputs.in_evals = 'dwi_evals.nii'
    >>> res = dti.run() # doctest: +SKIP


    """
    input_spec = EstimateResponseSHInputSpec
    output_spec = EstimateResponseSHOutputSpec

    def _run_interface(self, runtime):
        from dipy.core.gradients import GradientTable
        from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity
        from dipy.reconst.csdeconv import recursive_response, auto_response
        img = nb.load(self.inputs.in_file)
        imref = nb.four_to_three(img)[0]
        affine = img.affine
        if isdefined(self.inputs.in_mask):
            msk = np.asanyarray(nb.load(self.inputs.in_mask).dataobj)
            msk[msk > 0] = 1
            msk[msk < 0] = 0
        else:
            msk = np.ones(imref.shape)
        data = img.get_fdata(dtype=np.float32)
        gtab = self._get_gradient_table()
        evals = np.nan_to_num(nb.load(self.inputs.in_evals).dataobj)
        FA = np.nan_to_num(fractional_anisotropy(evals)) * msk
        indices = np.where(FA > self.inputs.fa_thresh)
        S0s = data[indices][:, np.nonzero(gtab.b0s_mask)[0]]
        S0 = np.mean(S0s)
        if self.inputs.auto:
            response, ratio = auto_response(gtab, data, roi_radius=self.inputs.roi_radius, fa_thr=self.inputs.fa_thresh)
            response = response[0].tolist() + [S0]
        elif self.inputs.recursive:
            MD = np.nan_to_num(mean_diffusivity(evals)) * msk
            indices = np.logical_or(FA >= 0.4, np.logical_and(FA >= 0.15, MD >= 0.0011))
            data = np.asanyarray(nb.load(self.inputs.in_file).dataobj)
            response = recursive_response(gtab, data, mask=indices, sh_order=8, peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8, convergence=0.001, parallel=True)
            ratio = abs(response[1] / response[0])
        else:
            lambdas = evals[indices]
            l01 = np.sort(np.mean(lambdas, axis=0))
            response = np.array([l01[-1], l01[-2], l01[-2], S0])
            ratio = abs(response[1] / response[0])
        if ratio > 0.25:
            IFLOGGER.warning('Estimated response is not prolate enough. Ratio=%0.3f.', ratio)
        elif ratio < 1e-05 or np.any(np.isnan(response)):
            response = np.array([0.0018, 0.00036, 0.00036, S0])
            IFLOGGER.warning('Estimated response is not valid, using a default one')
        else:
            IFLOGGER.info('Estimated response: %s', str(response[:3]))
        np.savetxt(op.abspath(self.inputs.response), response)
        wm_mask = np.zeros_like(FA)
        wm_mask[indices] = 1
        nb.Nifti1Image(wm_mask.astype(np.uint8), affine, None).to_filename(op.abspath(self.inputs.out_mask))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['response'] = op.abspath(self.inputs.response)
        outputs['out_mask'] = op.abspath(self.inputs.out_mask)
        return outputs