import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Edge3InputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dedge3', argstr='-input %s', position=0, mandatory=True, exists=True, copyfile=False)
    out_file = File(desc='output image file name', position=-1, argstr='-prefix %s')
    datum = traits.Enum('byte', 'short', 'float', argstr='-datum %s', desc="specify data type for output. Valid types are 'byte', 'short' and 'float'.")
    fscale = traits.Bool(desc='Force scaling of the output to the maximum integer range.', argstr='-fscale', xor=['gscale', 'nscale', 'scale_floats'])
    gscale = traits.Bool(desc="Same as '-fscale', but also forces each output sub-brick to to get the same scaling factor.", argstr='-gscale', xor=['fscale', 'nscale', 'scale_floats'])
    nscale = traits.Bool(desc="Don't do any scaling on output to byte or short datasets.", argstr='-nscale', xor=['fscale', 'gscale', 'scale_floats'])
    scale_floats = traits.Float(desc='Multiply input by VAL, but only if the input datum is float. This is needed when the input dataset has a small range, like 0 to 2.0 for instance. With such a range, very few edges are detected due to what I suspect to be truncation problems. Multiplying such a dataset by 10000 fixes the problem and the scaling is undone at the output.', argstr='-scale_floats %f', xor=['fscale', 'gscale', 'nscale'])
    verbose = traits.Bool(desc='Print out some information along the way.', argstr='-verbose')