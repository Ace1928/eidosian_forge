import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorrMapOutputSpec(TraitedSpec):
    mean_file = File()
    zmean = File()
    qmean = File()
    pmean = File()
    absolute_threshold = File()
    var_absolute_threshold = File()
    var_absolute_threshold_normalize = File()
    correlation_maps = File()
    correlation_maps_masked = File()
    average_expr = File()
    average_expr_nonzero = File()
    sum_expr = File()
    histogram = File()