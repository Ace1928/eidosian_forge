import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class LoganRefInputSpec(GLMFitInputSpec):
    logan = traits.Tuple(File(exists=True), File(exists=True), traits.Float, mandatory=True, argstr='--logan %s %s %g', desc='RefTac TimeSec tstar   : perform Logan kinetic modeling')