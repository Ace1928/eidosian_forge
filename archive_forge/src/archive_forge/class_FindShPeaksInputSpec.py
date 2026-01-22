import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class FindShPeaksInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='the input image of SH coefficients.')
    directions_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='the set of directions to use as seeds for the peak finding')
    peaks_image = File(exists=True, argstr='-peaks %s', desc='the program will try to find the peaks that most closely match those in the image provided')
    num_peaks = traits.Int(argstr='-num %s', desc='the number of peaks to extract (default is 3)')
    peak_directions = traits.List(traits.Float, argstr='-direction %s', sep=' ', minlen=2, maxlen=2, desc='phi theta.  the direction of a peak to estimate. The algorithm will attempt to find the same number of peaks as have been specified using this option  phi: the azimuthal angle of the direction (in degrees). theta: the elevation angle of the direction (in degrees, from the vertical z-axis)')
    peak_threshold = traits.Float(argstr='-threshold %s', desc='only peak amplitudes greater than the threshold will be considered')
    display_info = traits.Bool(argstr='-info', desc='Display information messages.')
    quiet_display = traits.Bool(argstr='-quiet', desc='do not display information messages or progress status.')
    display_debug = traits.Bool(argstr='-debug', desc='Display debugging messages.')
    out_file = File(name_template='%s_peak_dirs.mif', keep_extension=False, argstr='%s', hash_files=False, position=-1, desc='the output image. Each volume corresponds to the x, y & z component of each peak direction vector in turn', name_source=['in_file'])