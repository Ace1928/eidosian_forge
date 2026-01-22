import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class FixTopologyInputSpec(FSTraitedSpec):
    in_orig = File(exists=True, mandatory=True, desc='Undocumented input file <hemisphere>.orig')
    in_inflated = File(exists=True, mandatory=True, desc='Undocumented input file <hemisphere>.inflated')
    in_brain = File(exists=True, mandatory=True, desc='Implicit input brain.mgz')
    in_wm = File(exists=True, mandatory=True, desc='Implicit input wm.mgz')
    hemisphere = traits.String(position=-1, argstr='%s', mandatory=True, desc='Hemisphere being processed')
    subject_id = traits.String('subject_id', position=-2, argstr='%s', mandatory=True, usedefault=True, desc='Subject being processed')
    copy_inputs = traits.Bool(mandatory=True, desc='If running as a node, set this to True ' + 'otherwise, the topology fixing will be done ' + 'in place.')
    seed = traits.Int(argstr='-seed %d', desc='Seed for setting random number generator')
    ga = traits.Bool(argstr='-ga', desc='No documentation. Direct questions to analysis-bugs@nmr.mgh.harvard.edu')
    mgz = traits.Bool(argstr='-mgz', desc='No documentation. Direct questions to analysis-bugs@nmr.mgh.harvard.edu')
    sphere = File(argstr='-sphere %s', desc='Sphere input file')