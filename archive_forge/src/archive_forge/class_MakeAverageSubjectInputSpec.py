import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MakeAverageSubjectInputSpec(FSTraitedSpec):
    subjects_ids = traits.List(traits.Str(), argstr='--subjects %s', desc='freesurfer subjects ids to average', mandatory=True, sep=' ')
    out_name = File('average', argstr='--out %s', desc='name for the average subject', usedefault=True)