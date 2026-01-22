import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class EstimateFOD(MRTrix3Base):
    """
    Estimate fibre orientation distributions from diffusion data using spherical deconvolution

    .. warning::

       The CSD algorithm does not work as intended, but fixing it in this interface could break
       existing workflows. This interface has been superseded by
       :py:class:`.ConstrainedSphericalDecomposition`.

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> fod = mrt.EstimateFOD()
    >>> fod.inputs.algorithm = 'msmt_csd'
    >>> fod.inputs.in_file = 'dwi.mif'
    >>> fod.inputs.wm_txt = 'wm.txt'
    >>> fod.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> fod.cmdline
    'dwi2fod -fslgrad bvecs bvals -lmax 8 msmt_csd dwi.mif wm.txt wm.mif gm.mif csf.mif'
    >>> fod.run()  # doctest: +SKIP
    """
    _cmd = 'dwi2fod'
    input_spec = EstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_odf'] = op.abspath(self.inputs.wm_odf)
        if isdefined(self.inputs.gm_odf):
            outputs['gm_odf'] = op.abspath(self.inputs.gm_odf)
        if isdefined(self.inputs.csf_odf):
            outputs['csf_odf'] = op.abspath(self.inputs.csf_odf)
        if isdefined(self.inputs.predicted_signal):
            if self.inputs.algorithm != 'msmt_csd':
                raise Exception("'predicted_signal' option can only be used with the 'msmt_csd' algorithm")
            outputs['predicted_signal'] = op.abspath(self.inputs.predicted_signal)
        return outputs