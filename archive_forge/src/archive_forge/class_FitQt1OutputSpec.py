from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitQt1OutputSpec(TraitedSpec):
    """Output Spec for FitQt1."""
    t1map_file = File(desc='Filename of the estimated output T1 map (in ms)')
    m0map_file = File(desc='Filename of the m0 map')
    desc = 'Filename of the estimated output multi-parameter map'
    mcmap_file = File(desc=desc)
    comp_file = File(desc='Filename of the estimated multi-component T1 map.')
    desc = 'Filename of the error map (symmetric matrix, [Diag,OffDiag])'
    error_file = File(desc=desc)
    syn_file = File(desc='Filename of the synthetic ASL data')
    res_file = File(desc='Filename of the model fit residuals')