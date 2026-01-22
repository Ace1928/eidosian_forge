from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitDwi(NiftyFitCommand):
    """Interface for executable fit_dwi from Niftyfit platform.

    Use NiftyFit to perform diffusion model fitting.

    Diffusion-weighted MR Fitting.
    Fits DWI parameter maps to multi-shell, multi-directional data.

    `Source code <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit-Release>`_

    Examples
    --------

    >>> from nipype.interfaces import niftyfit
    >>> fit_dwi = niftyfit.FitDwi(dti_flag=True)
    >>> fit_dwi.inputs.source_file = 'dwi.nii.gz'
    >>> fit_dwi.inputs.bvec_file = 'bvecs'
    >>> fit_dwi.inputs.bval_file = 'bvals'
    >>> fit_dwi.inputs.rgbmap_file = 'rgb.nii.gz'
    >>> fit_dwi.cmdline
    'fit_dwi -source dwi.nii.gz -bval bvals -bvec bvecs -dti -error dwi_error.nii.gz -famap dwi_famap.nii.gz -mcout dwi_mcout.txt -mdmap dwi_mdmap.nii.gz -nodiff dwi_no_diff.nii.gz -res dwi_resmap.nii.gz -rgbmap rgb.nii.gz -syn dwi_syn.nii.gz -tenmap2 dwi_tenmap2.nii.gz -v1map dwi_v1map.nii.gz'

    """
    _cmd = get_custom_path('fit_dwi', env_dir='NIFTYFITDIR')
    input_spec = FitDwiInputSpec
    output_spec = FitDwiOutputSpec
    _suffix = '_fit_dwi'

    def _format_arg(self, name, trait_spec, value):
        if name == 'tenmap_file' and self.inputs.ten_type != 'diag-off-diag':
            return ''
        if name == 'tenmap2_file' and self.inputs.ten_type != 'lower-tri':
            return ''
        return super(FitDwi, self)._format_arg(name, trait_spec, value)