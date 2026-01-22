from ..base import File, TraitedSpec, traits, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
Interface for executable fit_asl from Niftyfit platform.

    Use NiftyFit to perform ASL fitting.

    ASL fitting routines (following EU Cost Action White Paper recommendations)
    Fits Cerebral Blood Flow maps in the first instance.

    `Source code <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit-Release>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyfit
    >>> node = niftyfit.FitAsl()
    >>> node.inputs.source_file = 'asl.nii.gz'
    >>> node.cmdline
    'fit_asl -source asl.nii.gz -cbf asl_cbf.nii.gz -error asl_error.nii.gz -syn asl_syn.nii.gz'

    