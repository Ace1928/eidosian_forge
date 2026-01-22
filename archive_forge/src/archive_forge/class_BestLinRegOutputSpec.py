import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BestLinRegOutputSpec(TraitedSpec):
    output_xfm = File(desc='output xfm file', exists=True)
    output_mnc = File(desc='output mnc file', exists=True)