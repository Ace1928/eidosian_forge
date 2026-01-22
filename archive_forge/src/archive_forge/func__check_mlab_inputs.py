import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def _check_mlab_inputs(self):
    if not isdefined(self.inputs.matlab_cmd) and self._matlab_cmd:
        self.inputs.matlab_cmd = self._matlab_cmd
    if not isdefined(self.inputs.paths) and self._paths:
        self.inputs.paths = self._paths
    if not isdefined(self.inputs.use_mcr) and self._use_mcr:
        self.inputs.use_mcr = self._use_mcr