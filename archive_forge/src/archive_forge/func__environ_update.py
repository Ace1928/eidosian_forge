import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
def _environ_update(self):
    refdir = self.inputs.reference_dir
    target = self.inputs.target
    self.inputs.environ['MPR2MNI305_TARGET'] = target
    self.inputs.environ['REFDIR'] = refdir