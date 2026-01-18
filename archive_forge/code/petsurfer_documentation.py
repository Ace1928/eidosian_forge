import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
Perform Logan reference kinetic modeling.
    Examples
    --------
    >>> logan = LoganRef()
    >>> logan.inputs.in_file = 'tac.nii'
    >>> logan.inputs.logan = ('ref_tac.dat', 'timing.dat', 2600)
    >>> logan.inputs.glm_dir = 'logan'
    >>> logan.cmdline
    'mri_glmfit --glmdir logan --y tac.nii --logan ref_tac.dat timing.dat 2600'
    