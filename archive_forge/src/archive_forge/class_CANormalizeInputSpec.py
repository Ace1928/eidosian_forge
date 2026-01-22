import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class CANormalizeInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-4, desc='The input file for CANormalize')
    out_file = File(argstr='%s', position=-1, name_source=['in_file'], name_template='%s_norm', hash_files=False, keep_extension=True, desc='The output file for CANormalize')
    atlas = File(argstr='%s', exists=True, mandatory=True, position=-3, desc='The atlas file in gca format')
    transform = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='The transform file in lta format')
    mask = File(argstr='-mask %s', exists=True, desc='Specifies volume to use as mask')
    control_points = File(argstr='-c %s', desc='File name for the output control points')
    long_file = File(argstr='-long %s', desc='undocumented flag used in longitudinal processing')