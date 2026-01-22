import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class RandomiseInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='4D input file', argstr='-i %s', position=0, mandatory=True)
    base_name = traits.Str('randomise', desc='the rootname that all generated files will have', argstr='-o "%s"', position=1, usedefault=True)
    design_mat = File(exists=True, desc='design matrix file', argstr='-d %s', position=2)
    tcon = File(exists=True, desc='t contrasts file', argstr='-t %s', position=3)
    fcon = File(exists=True, desc='f contrasts file', argstr='-f %s')
    mask = File(exists=True, desc='mask image', argstr='-m %s')
    x_block_labels = File(exists=True, desc='exchangeability block labels file', argstr='-e %s')
    demean = traits.Bool(desc='demean data temporally before model fitting', argstr='-D')
    one_sample_group_mean = traits.Bool(desc='perform 1-sample group-mean test instead of generic permutation test', argstr='-1')
    show_total_perms = traits.Bool(desc='print out how many unique permutations would be generated and exit', argstr='-q')
    show_info_parallel_mode = traits.Bool(desc='print out information required for parallel mode and exit', argstr='-Q')
    vox_p_values = traits.Bool(desc='output voxelwise (corrected and uncorrected) p-value images', argstr='-x')
    tfce = traits.Bool(desc='carry out Threshold-Free Cluster Enhancement', argstr='-T')
    tfce2D = traits.Bool(desc='carry out Threshold-Free Cluster Enhancement with 2D optimisation', argstr='--T2')
    f_only = traits.Bool(desc='calculate f-statistics only', argstr='--fonly')
    raw_stats_imgs = traits.Bool(desc='output raw ( unpermuted ) statistic images', argstr='-R')
    p_vec_n_dist_files = traits.Bool(desc='output permutation vector and null distribution text files', argstr='-P')
    num_perm = traits.Int(argstr='-n %d', desc='number of permutations (default 5000, set to 0 for exhaustive)')
    seed = traits.Int(argstr='--seed=%d', desc='specific integer seed for random number generator')
    var_smooth = traits.Int(argstr='-v %d', desc='use variance smoothing (std is in mm)')
    c_thresh = traits.Float(argstr='-c %.1f', desc='carry out cluster-based thresholding')
    cm_thresh = traits.Float(argstr='-C %.1f', desc='carry out cluster-mass-based thresholding')
    f_c_thresh = traits.Float(argstr='-F %.2f', desc='carry out f cluster thresholding')
    f_cm_thresh = traits.Float(argstr='-S %.2f', desc='carry out f cluster-mass thresholding')
    tfce_H = traits.Float(argstr='--tfce_H=%.2f', desc='TFCE height parameter (default=2)')
    tfce_E = traits.Float(argstr='--tfce_E=%.2f', desc='TFCE extent parameter (default=0.5)')
    tfce_C = traits.Float(argstr='--tfce_C=%.2f', desc='TFCE connectivity (6 or 26; default=6)')