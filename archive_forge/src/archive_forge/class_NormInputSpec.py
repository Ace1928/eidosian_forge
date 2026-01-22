import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class NormInputSpec(CommandLineInputSpec):
    """

    Not implemented:

       -version         print version and exit
       -verbose         be verbose
       -noverbose       opposite of -verbose [default]
       -quiet           be quiet
       -noquiet         opposite of -quiet [default]
       -fake            do a dry run, (echo cmds only)
       -nofake          opposite of -fake [default]
    """
    input_file = File(desc='input file to normalise', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_norm.mnc')
    output_threshold_mask = File(desc='File in which to store the threshold mask.', argstr='-threshold_mask %s', name_source=['input_file'], hash_files=False, name_template='%s_norm_threshold_mask.mnc')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    mask = File(desc='Calculate the image normalisation within a mask.', argstr='-mask %s', exists=True)
    clamp = traits.Bool(desc='Force the output range between limits [default].', argstr='-clamp', usedefault=True, default_value=True)
    cutoff = traits.Range(low=0.0, high=100.0, desc='Cutoff value to use to calculate thresholds by a histogram PcT in %. [default: 0.01]', argstr='-cutoff %s')
    lower = traits.Float(desc='Lower real value to use.', argstr='-lower %s')
    upper = traits.Float(desc='Upper real value to use.', argstr='-upper %s')
    out_floor = traits.Float(desc='Output files maximum [default: 0]', argstr='-out_floor %s')
    out_ceil = traits.Float(desc='Output files minimum [default: 100]', argstr='-out_ceil %s')
    threshold = traits.Bool(desc='Threshold the image (set values below threshold_perc to -out_floor).', argstr='-threshold')
    threshold_perc = traits.Range(low=0.0, high=100.0, desc='Threshold percentage (0.1 == lower 10% of intensity range) [default: 0.1].', argstr='-threshold_perc %s')
    threshold_bmt = traits.Bool(desc='Use the resulting image BiModalT as the threshold.', argstr='-threshold_bmt')
    threshold_blur = traits.Float(desc='Blur FWHM for intensity edges then thresholding [default: 2].', argstr='-threshold_blur %s')