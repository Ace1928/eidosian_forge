import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class LocalBistatInputSpec(AFNICommandInputSpec):
    in_file1 = File(exists=True, mandatory=True, argstr='%s', position=-2, desc='Filename of the first image')
    in_file2 = File(exists=True, mandatory=True, argstr='%s', position=-1, desc='Filename of the second image')
    neighborhood = traits.Either(traits.Tuple(traits.Enum('SPHERE', 'RHDD', 'TOHD'), traits.Float()), traits.Tuple(traits.Enum('RECT'), traits.Tuple(traits.Float(), traits.Float(), traits.Float())), mandatory=True, desc="The region around each voxel that will be extracted for the statistics calculation. Possible regions are: 'SPHERE', 'RHDD' (rhombic dodecahedron), 'TOHD' (truncated octahedron) with a given radius in mm or 'RECT' (rectangular block) with dimensions to specify in mm.", argstr="-nbhd '%s(%s)'")
    _stat_names = ['pearson', 'spearman', 'quadrant', 'mutinfo', 'normuti', 'jointent', 'hellinger', 'crU', 'crM', 'crA', 'L2slope', 'L1slope', 'num', 'ALL']
    stat = InputMultiPath(traits.Enum(_stat_names), mandatory=True, desc="Statistics to compute. Possible names are:\n\n  * pearson  = Pearson correlation coefficient\n  * spearman = Spearman correlation coefficient\n  * quadrant = Quadrant correlation coefficient\n  * mutinfo  = Mutual Information\n  * normuti  = Normalized Mutual Information\n  * jointent = Joint entropy\n  * hellinger= Hellinger metric\n  * crU      = Correlation ratio (Unsymmetric)\n  * crM      = Correlation ratio (symmetrized by Multiplication)\n  * crA      = Correlation ratio (symmetrized by Addition)\n  * L2slope  = slope of least-squares (L2) linear regression of\n               the data from dataset1 vs. the dataset2\n               (i.e., d2 = a + b*d1 ==> this is 'b')\n  * L1slope  = slope of least-absolute-sum (L1) linear\n               regression of the data from dataset1 vs.\n               the dataset2\n  * num      = number of the values in the region:\n               with the use of -mask or -automask,\n               the size of the region around any given\n               voxel will vary; this option lets you\n               map that size.\n  * ALL      = all of the above, in that order\n\nMore than one option can be used.", argstr='-stat %s...')
    mask_file = File(exists=True, desc='mask image file name. Voxels NOT in the mask will not be used in the neighborhood of any voxel. Also, a voxel NOT in the mask will have its statistic(s) computed as zero (0).', argstr='-mask %s')
    automask = traits.Bool(desc='Compute the mask as in program 3dAutomask.', argstr='-automask', xor=['weight_file'])
    weight_file = File(exists=True, desc="File name of an image to use as a weight.  Only applies to 'pearson' statistics.", argstr='-weight %s', xor=['automask'])
    out_file = File(desc='Output dataset.', argstr='-prefix %s', name_source='in_file1', name_template='%s_bistat', keep_extension=True, position=0)