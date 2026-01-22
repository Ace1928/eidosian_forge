import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class MRTM2(GLMFit):
    """Perform MRTM2 kinetic modeling.
    Examples
    --------
    >>> mrtm2 = MRTM2()
    >>> mrtm2.inputs.in_file = 'tac.nii'
    >>> mrtm2.inputs.mrtm2 = ('ref_tac.dat', 'timing.dat', 0.07872)
    >>> mrtm2.inputs.glm_dir = 'mrtm2'
    >>> mrtm2.cmdline
    'mri_glmfit --glmdir mrtm2 --y tac.nii --mrtm2 ref_tac.dat timing.dat 0.078720'
    """
    input_spec = MRTM2InputSpec