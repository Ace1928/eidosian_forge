import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class ConcatenateInputSpec(FSTraitedSpec):
    in_files = InputMultiPath(File(exists=True), desc='Individual volumes to be concatenated', argstr='--i %s...', mandatory=True)
    concatenated_file = File(desc='Output volume', argstr='--o %s', genfile=True)
    sign = traits.Enum('abs', 'pos', 'neg', argstr='--%s', desc='Take only pos or neg voxles from input, or take abs')
    stats = traits.Enum('sum', 'var', 'std', 'max', 'min', 'mean', argstr='--%s', desc='Compute the sum, var, std, max, min or mean of the input volumes')
    paired_stats = traits.Enum('sum', 'avg', 'diff', 'diff-norm', 'diff-norm1', 'diff-norm2', argstr='--paired-%s', desc='Compute paired sum, avg, or diff')
    gmean = traits.Int(argstr='--gmean %d', desc='create matrix to average Ng groups, Nper=Ntot/Ng')
    mean_div_n = traits.Bool(argstr='--mean-div-n', desc='compute mean/nframes (good for var)')
    multiply_by = traits.Float(argstr='--mul %f', desc='Multiply input volume by some amount')
    add_val = traits.Float(argstr='--add %f', desc='Add some amount to the input volume')
    multiply_matrix_file = File(exists=True, argstr='--mtx %s', desc='Multiply input by an ascii matrix in file')
    combine = traits.Bool(argstr='--combine', desc='Combine non-zero values into single frame volume')
    keep_dtype = traits.Bool(argstr='--keep-datatype', desc='Keep voxelwise precision type (default is float')
    max_bonfcor = traits.Bool(argstr='--max-bonfcor', desc='Compute max and bonferroni correct (assumes -log10(ps))')
    max_index = traits.Bool(argstr='--max-index', desc='Compute the index of max voxel in concatenated volumes')
    mask_file = File(exists=True, argstr='--mask %s', desc='Mask input with a volume')
    vote = traits.Bool(argstr='--vote', desc='Most frequent value at each voxel and fraction of occurrences')
    sort = traits.Bool(argstr='--sort', desc='Sort each voxel by ascending frame value')