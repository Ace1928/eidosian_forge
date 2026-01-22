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
class SPMCommandInputSpec(BaseInterfaceInputSpec):
    matlab_cmd = traits.Str(desc='matlab command to use')
    paths = InputMultiPath(Directory(), desc='Paths to add to matlabpath')
    mfile = traits.Bool(True, desc='Run m-code using m-file', usedefault=True)
    use_mcr = traits.Bool(desc='Run m-code using SPM MCR')
    use_v8struct = traits.Bool(True, min_ver='8', usedefault=True, desc='Generate SPM8 and higher compatible jobs')