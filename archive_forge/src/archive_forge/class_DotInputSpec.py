import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class DotInputSpec(AFNICommandInputSpec):
    in_files = traits.List(File(), desc='list of input files, possibly with subbrick selectors', argstr='%s ...', position=-2)
    out_file = File(desc='collect output to a file', argstr=' |& tee %s', position=-1)
    mask = File(desc='Use this dataset as a mask', argstr='-mask %s')
    mrange = traits.Tuple((traits.Float(), traits.Float()), desc="Means to further restrict the voxels from 'mset' so thatonly those mask values within this range (inclusive) willbe used.", argstr='-mrange %s %s')
    demean = traits.Bool(desc='Remove the mean from each volume prior to computing the correlation', argstr='-demean')
    docor = traits.Bool(desc='Return the correlation coefficient (default).', argstr='-docor')
    dodot = traits.Bool(desc='Return the dot product (unscaled).', argstr='-dodot')
    docoef = traits.Bool(desc='Return the least square fit coefficients {{a,b}} so that dset2 is approximately a + b\\*dset1', argstr='-docoef')
    dosums = traits.Bool(desc='Return the 6 numbers xbar=<x> ybar=<y> <(x-xbar)^2> <(y-ybar)^2> <(x-xbar)(y-ybar)> and the correlation coefficient.', argstr='-dosums')
    dodice = traits.Bool(desc='Return the Dice coefficient (the Sorensen-Dice index).', argstr='-dodice')
    doeta2 = traits.Bool(desc='Return eta-squared (Cohen, NeuroImage 2008).', argstr='-doeta2')
    full = traits.Bool(desc='Compute the whole matrix. A waste of time, but handy for parsing.', argstr='-full')
    show_labels = traits.Bool(desc='Print sub-brick labels to help identify what is being correlated. This option is useful whenyou have more than 2 sub-bricks at input.', argstr='-show_labels')
    upper = traits.Bool(desc='Compute upper triangular matrix', argstr='-upper')