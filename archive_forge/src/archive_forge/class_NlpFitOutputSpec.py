import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class NlpFitOutputSpec(TraitedSpec):
    output_xfm = File(desc='output xfm file', exists=True)
    output_grid = File(desc='output grid file', exists=True)