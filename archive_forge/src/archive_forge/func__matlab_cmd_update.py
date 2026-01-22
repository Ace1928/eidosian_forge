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
def _matlab_cmd_update(self):
    self.mlab = MatlabCommand(matlab_cmd=self.inputs.matlab_cmd, mfile=self.inputs.mfile, paths=self.inputs.paths, resource_monitor=False)
    self.mlab.inputs.script_file = 'pyscript_%s.m' % self.__class__.__name__.split('.')[-1].lower()
    if isdefined(self.inputs.use_mcr) and self.inputs.use_mcr:
        self.mlab.inputs.nodesktop = Undefined
        self.mlab.inputs.nosplash = Undefined
        self.mlab.inputs.single_comp_thread = Undefined
        self.mlab.inputs.uses_mcr = True
        self.mlab.inputs.mfile = True