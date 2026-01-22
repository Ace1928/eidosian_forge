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
class CARegisterInputSpec(FSTraitedSpecOpenMP):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-3, desc='The input volume for CARegister')
    out_file = File(argstr='%s', position=-1, genfile=True, desc='The output volume for CARegister')
    template = File(argstr='%s', exists=True, position=-2, desc='The template file in gca format')
    mask = File(argstr='-mask %s', exists=True, desc='Specifies volume to use as mask')
    invert_and_save = traits.Bool(argstr='-invert-and-save', position=-4, desc='Invert and save the .m3z multi-dimensional talaraich transform to x, y, and z .mgz files')
    no_big_ventricles = traits.Bool(argstr='-nobigventricles', desc='No big ventricles')
    transform = File(argstr='-T %s', exists=True, desc='Specifies transform in lta format')
    align = traits.String(argstr='-align-%s', desc='Specifies when to perform alignment')
    levels = traits.Int(argstr='-levels %d', desc='defines how many surrounding voxels will be used in interpolations, default is 6')
    A = traits.Int(argstr='-A %d', desc='undocumented flag used in longitudinal processing')
    l_files = InputMultiPath(File(exists=False), argstr='-l %s', desc='undocumented flag used in longitudinal processing')