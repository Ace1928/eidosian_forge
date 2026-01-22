import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CenterMassOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output file')
    cm_file = File(desc='file with the center of mass coordinates')
    cm = traits.List(traits.Tuple(traits.Float(), traits.Float(), traits.Float()), desc='center of mass')