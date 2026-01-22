import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class QwarpPlusMinusInputSpec(QwarpInputSpec):
    source_file = File(desc='Source image (opposite phase encoding direction than base image)', argstr='-source %s', exists=True, deprecated='1.1.2', new_name='in_file', copyfile=False)
    out_file = File('Qwarp.nii.gz', argstr='-prefix %s', position=0, usedefault=True, desc='Output file')
    plusminus = traits.Bool(True, usedefault=True, position=1, desc="Normally, the warp displacements dis(x) are defined to matchbase(x) to source(x+dis(x)).  With this option, the matchis between base(x-dis(x)) and source(x+dis(x)) -- the twoimages 'meet in the middle'. For more info, view Qwarp` interface", argstr='-plusminus', xor=['duplo', 'allsave', 'iwarp'])