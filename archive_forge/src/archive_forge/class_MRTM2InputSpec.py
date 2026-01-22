import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class MRTM2InputSpec(GLMFitInputSpec):
    mrtm2 = traits.Tuple(File(exists=True), File(exists=True), traits.Float, mandatory=True, argstr='--mrtm2 %s %s %f', desc='RefTac TimeSec k2prime : perform MRTM2 kinetic modeling')