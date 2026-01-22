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
class CALabelInputSpec(FSTraitedSpecOpenMP):
    in_file = File(argstr='%s', position=-4, mandatory=True, exists=True, desc='Input volume for CALabel')
    out_file = File(argstr='%s', position=-1, mandatory=True, exists=False, desc='Output file for CALabel')
    transform = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Input transform for CALabel')
    template = File(argstr='%s', position=-2, mandatory=True, exists=True, desc='Input template for CALabel')
    in_vol = File(argstr='-r %s', exists=True, desc='set input volume')
    intensities = File(argstr='-r %s', exists=True, desc='input label intensities file(used in longitudinal processing)')
    no_big_ventricles = traits.Bool(argstr='-nobigventricles', desc='No big ventricles')
    align = traits.Bool(argstr='-align', desc='Align CALabel')
    prior = traits.Float(argstr='-prior %.1f', desc='Prior for CALabel')
    relabel_unlikely = traits.Tuple(traits.Int, traits.Float, argstr='-relabel_unlikely %d %.1f', desc='Reclassify voxels at least some std devs from the mean using some size Gaussian window')
    label = File(argstr='-l %s', exists=True, desc='Undocumented flag. Autorecon3 uses ../label/{hemisphere}.cortex.label as input file')
    aseg = File(argstr='-aseg %s', exists=True, desc='Undocumented flag. Autorecon3 uses ../mri/aseg.presurf.mgz as input file')