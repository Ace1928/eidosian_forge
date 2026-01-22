import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BeastInputSpec(CommandLineInputSpec):
    """

    TODO:

    Command-specific options:
     -verbose:          Enable verbose output.
     -positive:         Specify mask of positive segmentation (inside mask) instead of the default mask.
     -output_selection: Specify file to output selected files.
     -count:            Specify file to output the patch count.
     -mask:             Specify a segmentation mask instead of the default mask.
     -no_mask:          Do not apply a segmentation mask. Perform the segmentation over the entire image.
     -no_positive:      Do not apply a positive mask.
    Generic options for all commands:
     -help:             Print summary of command-line options and abort
     -version:          Print version number of program and exit
    Copyright (C) 2011  Simon Fristed Eskildsen, Vladimir Fonov,
                Pierrick Coupe, Jose V. Manjon

    This program comes with ABSOLUTELY NO WARRANTY; for details type 'cat COPYING'.
    This is free software, and you are welcome to redistribute it under certain
    conditions; type 'cat COPYING' for details.

    Usage: mincbeast [options] <library dir> <input> <output>
           mincbeast -help

    Get this example to work?

    https://github.com/BIC-MNI/BEaST/blob/master/README.library


        2.3 Source the minc-toolkit (if installed):
        $ source /opt/minc/minc-toolkit-config.sh

        2.4 Generate library by running:
        $ beast_prepareADNIlib -flip <ADNI download directory> <BEaST library directory>
        Example:
        $ sudo beast_prepareADNIlib -flip Downloads/ADNI /opt/minc/share/beast-library-1.1

        3. Test the setup
        3.1 Normalize your data
        $ beast_normalize -modeldir /opt/minc/share/icbm152_model_09c input.mnc normal.mnc normal.xfm
        3.2 Run BEaST
        $ mincbeast /opt/minc/share/beast-library-1.1 normal.mnc brainmask.mnc -conf /opt/minc/share/beast-library-1.1/default.2mm.conf -same_res
    """
    probability_map = traits.Bool(desc='Output the probability map instead of crisp mask.', argstr='-probability')
    flip_images = traits.Bool(desc='Flip images around the mid-sagittal plane to increase patch count.', argstr='-flip')
    load_moments = traits.Bool(desc='Do not calculate moments instead use precalculatedlibrary moments. (for optimization purposes)', argstr='-load_moments')
    fill_holes = traits.Bool(desc='Fill holes in the binary output.', argstr='-fill')
    median_filter = traits.Bool(desc='Apply a median filter on the probability map.', argstr='-median')
    nlm_filter = traits.Bool(desc='Apply an NLM filter on the probability map (experimental).', argstr='-nlm_filter')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    configuration_file = File(desc='Specify configuration file.', argstr='-configuration %s')
    voxel_size = traits.Int(4, usedefault=True, desc='Specify voxel size for calculations (4, 2, or 1).Default value: 4. Assumes no multiscale. Use configurationfile for multiscale.', argstr='-voxel_size %s')
    abspath = traits.Bool(desc='File paths in the library are absolute (default is relative to library root).', argstr='-abspath', usedefault=True, default_value=True)
    patch_size = traits.Int(1, usedefault=True, desc='Specify patch size for single scale approach. Default value: 1.', argstr='-patch_size %s')
    search_area = traits.Int(2, usedefault=True, desc='Specify size of search area for single scale approach. Default value: 2.', argstr='-search_area %s')
    confidence_level_alpha = traits.Float(0.5, usedefault=True, desc='Specify confidence level Alpha. Default value: 0.5', argstr='-alpha %s')
    smoothness_factor_beta = traits.Float(0.5, usedefault=True, desc='Specify smoothness factor Beta. Default value: 0.25', argstr='-beta %s')
    threshold_patch_selection = traits.Float(0.95, usedefault=True, desc='Specify threshold for patch selection. Default value: 0.95', argstr='-threshold %s')
    number_selected_images = traits.Int(20, usedefault=True, desc='Specify number of selected images. Default value: 20', argstr='-selection_num %s')
    same_resolution = traits.Bool(desc='Output final mask with the same resolution as input file.', argstr='-same_resolution')
    library_dir = Directory(desc='library directory', position=-3, argstr='%s', mandatory=True)
    input_file = File(desc='input file', position=-2, argstr='%s', mandatory=True)
    output_file = File(desc='output file', position=-1, argstr='%s', name_source=['input_file'], hash_files=False, name_template='%s_beast_mask.mnc')