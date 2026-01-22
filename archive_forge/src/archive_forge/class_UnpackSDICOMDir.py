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
class UnpackSDICOMDir(FSCommand):
    """Use unpacksdcmdir to convert dicom files

    Call unpacksdcmdir -help from the command line to see more information on
    using this command.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import UnpackSDICOMDir
    >>> unpack = UnpackSDICOMDir()
    >>> unpack.inputs.source_dir = '.'
    >>> unpack.inputs.output_dir = '.'
    >>> unpack.inputs.run_info = (5, 'mprage', 'nii', 'struct')
    >>> unpack.inputs.dir_structure = 'generic'
    >>> unpack.cmdline
    'unpacksdcmdir -generic -targ . -run 5 mprage nii struct -src .'
    """
    _cmd = 'unpacksdcmdir'
    input_spec = UnpackSDICOMDirInputSpec