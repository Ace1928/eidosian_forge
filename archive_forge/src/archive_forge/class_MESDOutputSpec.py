import os
from ...utils.filemanip import split_filename
from ..base import (
class MESDOutputSpec(TraitedSpec):
    mesd_data = File(exists=True, desc='MESD data')