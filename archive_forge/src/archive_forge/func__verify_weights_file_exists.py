import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
def _verify_weights_file_exists(self):
    if not os.path.exists(os.path.abspath(self.inputs.weight_file)):
        raise traits.TraitError('MS_LDA: use_weights must accompany an existing weights file')