import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ImageMeantsInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='input file for computing the average timeseries', argstr='-i %s', position=0, mandatory=True)
    out_file = File(desc='name of output text matrix', argstr='-o %s', genfile=True, hash_files=False)
    mask = File(exists=True, desc='input 3D mask', argstr='-m %s')
    spatial_coord = traits.List(traits.Int, desc='<x y z>  requested spatial coordinate (instead of mask)', argstr='-c %s')
    use_mm = traits.Bool(desc='use mm instead of voxel coordinates (for -c option)', argstr='--usemm')
    show_all = traits.Bool(desc='show all voxel time series (within mask) instead of averaging', argstr='--showall')
    eig = traits.Bool(desc='calculate Eigenvariate(s) instead of mean (output will have 0 mean)', argstr='--eig')
    order = traits.Int(1, desc='select number of Eigenvariates', argstr='--order=%d', usedefault=True)
    nobin = traits.Bool(desc='do not binarise the mask for calculation of Eigenvariates', argstr='--no_bin')
    transpose = traits.Bool(desc='output results in transpose format (one row per voxel/mean)', argstr='--transpose')