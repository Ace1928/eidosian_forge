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
class SegmentCCInputSpec(FSTraitedSpec):
    in_file = File(argstr='-aseg %s', mandatory=True, exists=True, desc='Input aseg file to read from subjects directory')
    in_norm = File(mandatory=True, exists=True, desc='Required undocumented input {subject}/mri/norm.mgz')
    out_file = File(argstr='-o %s', exists=False, name_source=['in_file'], name_template='%s.auto.mgz', hash_files=False, keep_extension=False, desc='Filename to write aseg including CC')
    out_rotation = File(argstr='-lta %s', mandatory=True, exists=False, desc='Global filepath for writing rotation lta')
    subject_id = traits.String('subject_id', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='Subject name')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')