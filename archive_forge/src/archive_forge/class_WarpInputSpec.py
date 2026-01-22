import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class WarpInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dWarp', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_warp', desc='output image file name', argstr='-prefix %s', name_source='in_file', keep_extension=True)
    tta2mni = traits.Bool(desc='transform dataset from Talairach to MNI152', argstr='-tta2mni')
    mni2tta = traits.Bool(desc='transform dataset from MNI152 to Talaraich', argstr='-mni2tta')
    matparent = File(desc='apply transformation from 3dWarpDrive', argstr='-matparent %s', exists=True)
    oblique_parent = File(desc='Read in the oblique transformation matrix from an oblique dataset and make cardinal dataset oblique to match', argstr='-oblique_parent %s', exists=True)
    deoblique = traits.Bool(desc='transform dataset from oblique to cardinal', argstr='-deoblique')
    interp = traits.Enum(('linear', 'cubic', 'NN', 'quintic'), desc='spatial interpolation methods [default = linear]', argstr='-%s')
    gridset = File(desc='copy grid of specified dataset', argstr='-gridset %s', exists=True)
    newgrid = traits.Float(desc='specify grid of this size (mm)', argstr='-newgrid %f')
    zpad = traits.Int(desc='pad input dataset with N planes of zero on all sides.', argstr='-zpad %d')
    verbose = traits.Bool(desc='Print out some information along the way.', argstr='-verb')
    save_warp = traits.Bool(desc='save warp as .mat file', requires=['verbose'])