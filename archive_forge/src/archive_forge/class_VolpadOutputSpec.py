import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolpadOutputSpec(TraitedSpec):
    output_file = File(desc='output file', exists=True)