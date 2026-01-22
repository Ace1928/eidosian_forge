from nipype.interfaces.base import (
import os
class ThresholdScalarVolumeInputSpec(CommandLineInputSpec):
    InputVolume = File(position=-2, desc='Input volume', exists=True, argstr='%s')
    OutputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Thresholded input volume', argstr='%s')
    threshold = traits.Int(desc='Threshold value', argstr='--threshold %d')
    lower = traits.Int(desc='Lower threshold value', argstr='--lower %d')
    upper = traits.Int(desc='Upper threshold value', argstr='--upper %d')
    outsidevalue = traits.Int(desc='Set the voxels to this value if they fall outside the threshold range', argstr='--outsidevalue %d')
    thresholdtype = traits.Enum('Below', 'Above', 'Outside', desc='What kind of threshold to perform. If Outside is selected, uses Upper and Lower values. If Below is selected, uses the ThresholdValue, if Above is selected, uses the ThresholdValue.', argstr='--thresholdtype %s')