import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MotionOutliersInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, desc='unfiltered 4D image', argstr='-i %s')
    out_file = File(argstr='-o %s', name_source='in_file', name_template='%s_outliers.txt', keep_extension=True, desc='output outlier file name', hash_files=False)
    mask = File(exists=True, argstr='-m %s', desc='mask image for calculating metric')
    metric = traits.Enum('refrms', ['refrms', 'dvars', 'refmse', 'fd', 'fdrms'], argstr='--%s', desc='metrics: refrms - RMS intensity difference to reference volume as metric [default metric], refmse - Mean Square Error version of refrms (used in original version of fsl_motion_outliers), dvars - DVARS, fd - frame displacement, fdrms - FD with RMS matrix calculation')
    threshold = traits.Float(argstr='--thresh=%g', desc='specify absolute threshold value (otherwise use box-plot cutoff = P75 + 1.5*IQR)')
    no_motion_correction = traits.Bool(argstr='--nomoco', desc='do not run motion correction (assumed already done)')
    dummy = traits.Int(argstr='--dummy=%d', desc='number of dummy scans to delete (before running anything and creating EVs)')
    out_metric_values = File(argstr='-s %s', name_source='in_file', name_template='%s_metrics.txt', keep_extension=True, desc='output metric values (DVARS etc.) file name', hash_files=False)
    out_metric_plot = File(argstr='-p %s', name_source='in_file', name_template='%s_metrics.png', hash_files=False, keep_extension=True, desc='output metric values plot (DVARS etc.) file name')