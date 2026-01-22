from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifyModelOutputSpec(TraitedSpec):
    session_info = traits.Any(desc='Session info for level1designs')