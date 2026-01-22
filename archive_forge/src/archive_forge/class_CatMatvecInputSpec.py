import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CatMatvecInputSpec(AFNICommandInputSpec):
    in_file = traits.List(traits.Tuple(traits.Str(), traits.Str()), desc='list of tuples of mfiles and associated opkeys', mandatory=True, argstr='%s', position=-2)
    out_file = File(argstr=' > %s', name_template='%s_cat.aff12.1D', name_source='in_file', keep_extension=False, desc='File to write concattenated matvecs to', position=-1, mandatory=True)
    matrix = traits.Bool(desc="indicates that the resulting matrix willbe written to outfile in the 'MATRIX(...)' format (FORM 3).This feature could be used, with clever scripting, to inputa matrix directly on the command line to program 3dWarp.", argstr='-MATRIX', xor=['oneline', 'fourxfour'])
    oneline = traits.Bool(desc='indicates that the resulting matrixwill simply be written as 12 numbers on one line.', argstr='-ONELINE', xor=['matrix', 'fourxfour'])
    fourxfour = traits.Bool(desc='Output matrix in augmented form (last row is 0 0 0 1)This option does not work with -MATRIX or -ONELINE', argstr='-4x4', xor=['matrix', 'oneline'])