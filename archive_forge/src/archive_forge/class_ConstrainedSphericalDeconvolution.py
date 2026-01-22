import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class ConstrainedSphericalDeconvolution(EstimateFOD):
    """
    Estimate fibre orientation distributions from diffusion data using spherical deconvolution

    This interface supersedes :py:class:`.EstimateFOD`.
    The old interface has contained a bug when using the CSD algorithm as opposed to the MSMT CSD
    algorithm, but fixing it could potentially break existing workflows. The new interface works
    the same, but does not populate the following inputs by default:

    * ``gm_odf``
    * ``csf_odf``
    * ``max_sh``

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> fod = mrt.ConstrainedSphericalDeconvolution()
    >>> fod.inputs.algorithm = 'csd'
    >>> fod.inputs.in_file = 'dwi.mif'
    >>> fod.inputs.wm_txt = 'wm.txt'
    >>> fod.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> fod.cmdline
    'dwi2fod -fslgrad bvecs bvals csd dwi.mif wm.txt wm.mif'
    >>> fod.run()  # doctest: +SKIP
    """
    input_spec = ConstrainedSphericalDeconvolutionInputSpec