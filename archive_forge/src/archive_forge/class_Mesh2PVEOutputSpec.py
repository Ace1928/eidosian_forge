import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class Mesh2PVEOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')