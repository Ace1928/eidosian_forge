import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MakeSurfacesInputSpec(FSTraitedSpec):
    hemisphere = traits.Enum('lh', 'rh', position=-1, argstr='%s', mandatory=True, desc='Hemisphere being processed')
    subject_id = traits.String('subject_id', usedefault=True, position=-2, argstr='%s', mandatory=True, desc='Subject being processed')
    in_orig = File(exists=True, mandatory=True, argstr='-orig %s', desc='Implicit input file <hemisphere>.orig')
    in_wm = File(exists=True, mandatory=True, desc='Implicit input file wm.mgz')
    in_filled = File(exists=True, mandatory=True, desc='Implicit input file filled.mgz')
    in_white = File(exists=True, desc='Implicit input that is sometimes used')
    in_label = File(exists=True, xor=['noaparc'], desc='Implicit input label/<hemisphere>.aparc.annot')
    orig_white = File(argstr='-orig_white %s', exists=True, desc='Specify a white surface to start with')
    orig_pial = File(argstr='-orig_pial %s', exists=True, requires=['in_label'], desc='Specify a pial surface to start with')
    fix_mtl = traits.Bool(argstr='-fix_mtl', desc='Undocumented flag')
    no_white = traits.Bool(argstr='-nowhite', desc='Undocumented flag')
    white_only = traits.Bool(argstr='-whiteonly', desc='Undocumented flag')
    in_aseg = File(argstr='-aseg %s', exists=True, desc='Input segmentation file')
    in_T1 = File(argstr='-T1 %s', exists=True, desc='Input brain or T1 file')
    mgz = traits.Bool(argstr='-mgz', desc='No documentation. Direct questions to analysis-bugs@nmr.mgh.harvard.edu')
    noaparc = traits.Bool(argstr='-noaparc', xor=['in_label'], desc='No documentation. Direct questions to analysis-bugs@nmr.mgh.harvard.edu')
    maximum = traits.Float(argstr='-max %.1f', desc='No documentation (used for longitudinal processing)')
    longitudinal = traits.Bool(argstr='-long', desc='No documentation (used for longitudinal processing)')
    white = traits.String(argstr='-white %s', desc='White surface name')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')