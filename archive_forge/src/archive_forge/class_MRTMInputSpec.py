import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class MRTMInputSpec(GLMFitInputSpec):
    mrtm1 = traits.Tuple(File(exists=True), File(exists=True), mandatory=True, argstr='--mrtm1 %s %s', desc='RefTac TimeSec : perform MRTM1 kinetic modeling')