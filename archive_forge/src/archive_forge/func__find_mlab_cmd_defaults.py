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
def _find_mlab_cmd_defaults(self):
    if self._use_mcr or 'FORCE_SPMMCR' in os.environ:
        self._use_mcr = True
        if self._matlab_cmd is None:
            try:
                self._matlab_cmd = os.environ['SPMMCRCMD']
            except KeyError:
                pass