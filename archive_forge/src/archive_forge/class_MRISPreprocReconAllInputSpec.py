import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class MRISPreprocReconAllInputSpec(MRISPreprocInputSpec):
    surf_measure_file = File(exists=True, argstr='--meas %s', xor=('surf_measure', 'surf_measure_file', 'surf_area'), desc='file necessary for surfmeas')
    surfreg_files = InputMultiPath(File(exists=True), argstr='--surfreg %s', requires=['lh_surfreg_target', 'rh_surfreg_target'], desc='lh and rh input surface registration files')
    lh_surfreg_target = File(desc='Implicit target surface registration file', requires=['surfreg_files'])
    rh_surfreg_target = File(desc='Implicit target surface registration file', requires=['surfreg_files'])
    subject_id = traits.String('subject_id', argstr='--s %s', usedefault=True, xor=('subjects', 'fsgd_file', 'subject_file', 'subject_id'), desc='subject from whom measures are calculated')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True this will copy some implicit inputs to the node directory.')