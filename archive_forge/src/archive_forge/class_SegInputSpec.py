import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class SegInputSpec(CommandLineInputSpec):
    in_file = File(desc='ANAT is the volume to segment', argstr='-anat %s', position=-1, mandatory=True, exists=True, copyfile=True)
    mask = traits.Either(traits.Enum('AUTO'), File(exists=True), desc='only non-zero voxels in mask are analyzed. mask can either be a dataset or the string "AUTO" which would use AFNI\'s automask function to create the mask.', argstr='-mask %s', position=-2, mandatory=True)
    blur_meth = traits.Enum('BFT', 'BIM', argstr='-blur_meth %s', desc='set the blurring method for bias field estimation')
    bias_fwhm = traits.Float(desc='The amount of blurring used when estimating the field bias with the Wells method', argstr='-bias_fwhm %f')
    classes = Str(desc='CLASS_STRING is a semicolon delimited string of class labels', argstr='-classes %s')
    bmrf = traits.Float(desc='Weighting factor controlling spatial homogeneity of the classifications', argstr='-bmrf %f')
    bias_classes = Str(desc='A semicolon delimited string of classes that contribute to the estimation of the bias field', argstr='-bias_classes %s')
    prefix = Str(desc='the prefix for the output folder containing all output volumes', argstr='-prefix %s')
    mixfrac = Str(desc='MIXFRAC sets up the volume-wide (within mask) tissue fractions while initializing the segmentation (see IGNORE for exception)', argstr='-mixfrac %s')
    mixfloor = traits.Float(desc="Set the minimum value for any class's mixing fraction", argstr='-mixfloor %f')
    main_N = traits.Int(desc='Number of iterations to perform.', argstr='-main_N %d')