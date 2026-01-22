import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PlotTimeSeriesInputSpec(FSLCommandInputSpec):
    in_file = traits.Either(File(exists=True), traits.List(File(exists=True)), mandatory=True, argstr='%s', position=1, desc='file or list of files with columns of timecourse information')
    plot_start = traits.Int(argstr='--start=%d', xor=('plot_range',), desc='first column from in-file to plot')
    plot_finish = traits.Int(argstr='--finish=%d', xor=('plot_range',), desc='final column from in-file to plot')
    plot_range = traits.Tuple(traits.Int, traits.Int, argstr='%s', xor=('plot_start', 'plot_finish'), desc='first and last columns from the in-file to plot')
    title = traits.Str(argstr='%s', desc='plot title')
    legend_file = File(exists=True, argstr='--legend=%s', desc='legend file')
    labels = traits.Either(traits.Str, traits.List(traits.Str), argstr='%s', desc='label or list of labels')
    y_min = traits.Float(argstr='--ymin=%.2f', desc='minimum y value', xor=('y_range',))
    y_max = traits.Float(argstr='--ymax=%.2f', desc='maximum y value', xor=('y_range',))
    y_range = traits.Tuple(traits.Float, traits.Float, argstr='%s', xor=('y_min', 'y_max'), desc='min and max y axis values')
    x_units = traits.Int(argstr='-u %d', usedefault=True, default_value=1, desc='scaling units for x-axis (between 1 and length of in file)')
    plot_size = traits.Tuple(traits.Int, traits.Int, argstr='%s', desc='plot image height and width')
    x_precision = traits.Int(argstr='--precision=%d', desc='precision of x-axis labels')
    sci_notation = traits.Bool(argstr='--sci', desc='switch on scientific notation')
    out_file = File(argstr='-o %s', genfile=True, desc='image to write', hash_files=False)