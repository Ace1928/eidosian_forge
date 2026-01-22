import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsExpandInputSpec(FSTraitedSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=-3, copyfile=False, desc='Surface to expand')
    distance = traits.Float(mandatory=True, argstr='%g', position=-2, desc='Distance in mm or fraction of cortical thickness')
    out_name = traits.Str('expanded', argstr='%s', position=-1, usedefault=True, desc='Output surface file. If no path, uses directory of ``in_file``. If no path AND missing "lh." or "rh.", derive from ``in_file``')
    thickness = traits.Bool(argstr='-thickness', desc='Expand by fraction of cortical thickness, not mm')
    thickness_name = traits.Str(argstr='-thickness_name %s', copyfile=False, desc='Name of thickness file (implicit: "thickness")\nIf no path, uses directory of ``in_file``\nIf no path AND missing "lh." or "rh.", derive from `in_file`')
    pial = traits.Str(argstr='-pial %s', copyfile=False, desc='Name of pial file (implicit: "pial")\nIf no path, uses directory of ``in_file``\nIf no path AND missing "lh." or "rh.", derive from ``in_file``')
    sphere = traits.Str('sphere', copyfile=False, usedefault=True, desc='WARNING: Do not change this trait')
    spring = traits.Float(argstr='-S %g', desc='Spring term (implicit: 0.05)')
    dt = traits.Float(argstr='-T %g', desc='dt (implicit: 0.25)')
    write_iterations = traits.Int(argstr='-W %d', desc='Write snapshots of expansion every N iterations')
    smooth_averages = traits.Int(argstr='-A %d', desc='Smooth surface with N iterations after expansion')
    nsurfaces = traits.Int(argstr='-N %d', desc='Number of surfacces to write during expansion')