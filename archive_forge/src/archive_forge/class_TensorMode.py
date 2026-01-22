import numpy as np
import nibabel as nb
from ... import logging
from ..base import TraitedSpec, File, isdefined
from .base import DipyDiffusionInterface, DipyBaseInterfaceInputSpec
class TensorMode(DipyDiffusionInterface):
    """
    Creates a map of the mode of the diffusion tensors given a set of
    diffusion-weighted images, as well as their associated b-values and
    b-vectors [1]_. Fits the diffusion tensors and calculates tensor mode
    with Dipy.

    Example
    -------
    >>> import nipype.interfaces.dipy as dipy
    >>> mode = dipy.TensorMode()
    >>> mode.inputs.in_file = 'diffusion.nii'
    >>> mode.inputs.in_bvec = 'bvecs'
    >>> mode.inputs.in_bval = 'bvals'
    >>> mode.run()                                   # doctest: +SKIP

    References
    ----------
    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.

    """
    input_spec = TensorModeInputSpec
    output_spec = TensorModeOutputSpec

    def _run_interface(self, runtime):
        from dipy.reconst import dti
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()
        affine = img.affine
        gtab = self._get_gradient_table()
        mask = data[..., 0] > 50
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask)
        mode_data = tenfit.mode
        img = nb.Nifti1Image(mode_data, affine)
        out_file = self._gen_filename('mode')
        nb.save(img, out_file)
        IFLOGGER.info('Tensor mode image saved as %s', out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('mode')
        return outputs