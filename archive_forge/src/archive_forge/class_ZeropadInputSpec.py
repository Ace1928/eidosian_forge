import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ZeropadInputSpec(AFNICommandInputSpec):
    in_files = File(desc='input dataset', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='zeropad', desc="output dataset prefix name (default 'zeropad')", argstr='-prefix %s')
    I = traits.Int(desc="adds 'n' planes of zero at the Inferior edge", argstr='-I %i', xor=['master'])
    S = traits.Int(desc="adds 'n' planes of zero at the Superior edge", argstr='-S %i', xor=['master'])
    A = traits.Int(desc="adds 'n' planes of zero at the Anterior edge", argstr='-A %i', xor=['master'])
    P = traits.Int(desc="adds 'n' planes of zero at the Posterior edge", argstr='-P %i', xor=['master'])
    L = traits.Int(desc="adds 'n' planes of zero at the Left edge", argstr='-L %i', xor=['master'])
    R = traits.Int(desc="adds 'n' planes of zero at the Right edge", argstr='-R %i', xor=['master'])
    z = traits.Int(desc="adds 'n' planes of zero on EACH of the dataset z-axis (slice-direction) faces", argstr='-z %i', xor=['master'])
    RL = traits.Int(desc='specify that planes should be added or cut symmetrically to make the resulting volume haveN slices in the right-left direction', argstr='-RL %i', xor=['master'])
    AP = traits.Int(desc='specify that planes should be added or cut symmetrically to make the resulting volume haveN slices in the anterior-posterior direction', argstr='-AP %i', xor=['master'])
    IS = traits.Int(desc='specify that planes should be added or cut symmetrically to make the resulting volume haveN slices in the inferior-superior direction', argstr='-IS %i', xor=['master'])
    mm = traits.Bool(desc="pad counts 'n' are in mm instead of slices, where each 'n' is an integer and at least 'n' mm of slices will be added/removed; e.g., n =  3 and slice thickness = 2.5 mm ==> 2 slices added", argstr='-mm', xor=['master'])
    master = File(desc="match the volume described in dataset 'mset', where mset must have the same orientation and grid spacing as dataset to be padded. the goal of -master is to make the output dataset from 3dZeropad match the spatial 'extents' of mset by adding or subtracting slices as needed. You can't use -I,-S,..., or -mm with -master", argstr='-master %s', xor=['I', 'S', 'A', 'P', 'L', 'R', 'z', 'RL', 'AP', 'IS', 'mm'])