from ..base import TraitedSpec, File, traits, CommandLineInputSpec, InputMultiPath
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class EMOutputSpec(TraitedSpec):
    """Output Spec for EM."""
    out_file = File(desc='Output segmentation')
    out_bc_file = File(desc='Output bias corrected image')
    out_outlier_file = File(desc='Output outlierness image')