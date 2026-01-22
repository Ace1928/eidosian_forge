import os
import re as regex
from ..base import (
class SkullfinderInputSpec(CommandLineInputSpec):
    inputMRIFile = File(mandatory=True, desc='input file', argstr='-i %s')
    inputMaskFile = File(mandatory=True, desc='A brain mask file, 8-bit image (0=non-brain, 255=brain)', argstr='-m %s')
    outputLabelFile = File(desc='output multi-colored label volume segmenting brain, scalp, inner skull & outer skull If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    verbosity = traits.Int(desc='verbosity', argstr='-v %d')
    lowerThreshold = traits.Int(desc='Lower threshold for segmentation', argstr='-l %d')
    upperThreshold = traits.Int(desc='Upper threshold for segmentation', argstr='-u %d')
    surfaceFilePrefix = traits.Str(desc='if specified, generate surface files for brain, skull, and scalp', argstr='-s %s')
    bgLabelValue = traits.Int(desc='background label value (0-255)', argstr='--bglabel %d')
    scalpLabelValue = traits.Int(desc='scalp label value (0-255)', argstr='--scalplabel %d')
    skullLabelValue = traits.Int(desc='skull label value (0-255)', argstr='--skulllabel %d')
    spaceLabelValue = traits.Int(desc='space label value (0-255)', argstr='--spacelabel %d')
    brainLabelValue = traits.Int(desc='brain label value (0-255)', argstr='--brainlabel %d')
    performFinalOpening = traits.Bool(desc='perform a final opening operation on the scalp mask', argstr='--finalOpening')