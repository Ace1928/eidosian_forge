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
class ParseDICOMDirInputSpec(FSTraitedSpec):
    dicom_dir = Directory(exists=True, argstr='--d %s', mandatory=True, desc='path to siemens dicom directory')
    dicom_info_file = File('dicominfo.txt', argstr='--o %s', usedefault=True, desc='file to which results are written')
    sortbyrun = traits.Bool(argstr='--sortbyrun', desc='assign run numbers')
    summarize = traits.Bool(argstr='--summarize', desc='only print out info for run leaders')