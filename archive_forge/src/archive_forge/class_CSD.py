import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class CSD(DipyDiffusionInterface):
    """
    Uses CSD [Tournier2007]_ to generate the fODF of DWIs. The interface uses
    :py:mod:`dipy`, as explained in `dipy's CSD example
    <http://nipy.org/dipy/examples_built/reconst_csd.html>`_.

    .. [Tournier2007] Tournier, J.D., et al. NeuroImage 2007.
      Robust determination of the fibre orientation distribution in diffusion
      MRI: Non-negativity constrained super-resolved spherical deconvolution


    Example
    -------

    >>> from nipype.interfaces import dipy as ndp
    >>> csd = ndp.CSD()
    >>> csd.inputs.in_file = '4d_dwi.nii'
    >>> csd.inputs.in_bval = 'bvals'
    >>> csd.inputs.in_bvec = 'bvecs'
    >>> res = csd.run() # doctest: +SKIP
    """
    input_spec = CSDInputSpec
    output_spec = CSDOutputSpec

    def _run_interface(self, runtime):
        from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
        from dipy.data import get_sphere
        import pickle as pickle
        import gzip
        img = nb.load(self.inputs.in_file)
        imref = nb.four_to_three(img)[0]
        if isdefined(self.inputs.in_mask):
            msk = np.asanyarray(nb.load(self.inputs.in_mask).dataobj)
        else:
            msk = np.ones(imref.shape)
        data = img.get_fdata(dtype=np.float32)
        gtab = self._get_gradient_table()
        resp_file = np.loadtxt(self.inputs.response)
        response = (np.array(resp_file[0:3]), resp_file[-1])
        ratio = response[0][1] / response[0][0]
        if abs(ratio - 0.2) > 0.1:
            IFLOGGER.warning('Estimated response is not prolate enough. Ratio=%0.3f.', ratio)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=self.inputs.sh_order)
        IFLOGGER.info('Fitting CSD model')
        csd_fit = csd_model.fit(data, msk)
        f = gzip.open(self._gen_filename('csdmodel', ext='.pklz'), 'wb')
        pickle.dump(csd_model, f, -1)
        f.close()
        if self.inputs.save_fods:
            sphere = get_sphere('symmetric724')
            fods = csd_fit.odf(sphere)
            nb.Nifti1Image(fods.astype(np.float32), img.affine, None).to_filename(self._gen_filename('fods'))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['model'] = self._gen_filename('csdmodel', ext='.pklz')
        if self.inputs.save_fods:
            outputs['out_fods'] = self._gen_filename('fods')
        return outputs