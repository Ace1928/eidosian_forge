import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpCatInputSpec(AFNICommandInputSpec):
    in_files = traits.List(traits.Either(File(), traits.Tuple(traits.Enum('IDENT', 'INV', 'SQRT', 'SQRTINV'), File())), desc='list of tuples of 3D warps and associated functions', mandatory=True, argstr='%s', position=-1)
    space = traits.String(desc='string to attach to the output dataset as its atlas space marker.', argstr='-space %s')
    inv_warp = traits.Bool(desc='invert the final warp before output', argstr='-iwarp')
    interp = traits.Enum('wsinc5', 'linear', 'quintic', desc='specify a different interpolation method than might be used for the warp', argstr='-interp %s', usedefault=True)
    expad = traits.Int(desc='Pad the nonlinear warps by the given number of voxels in all directions. The warp displacements are extended by linear extrapolation from the faces of the input grid..', argstr='-expad %d')
    out_file = File(name_template='%s_NwarpCat', desc='output image file name', argstr='-prefix %s', name_source='in_files')
    verb = traits.Bool(desc='be verbose', argstr='-verb')