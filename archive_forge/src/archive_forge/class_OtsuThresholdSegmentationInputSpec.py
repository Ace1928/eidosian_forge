from nipype.interfaces.base import (
import os
class OtsuThresholdSegmentationInputSpec(CommandLineInputSpec):
    brightObjects = traits.Bool(desc='Segmenting bright objects on a dark background or dark objects on a bright background.', argstr='--brightObjects ')
    numberOfBins = traits.Int(desc='This is an advanced parameter. The number of bins in the histogram used to model the probability mass function of the two intensity distributions. Small numbers of bins may result in a more conservative threshold. The default should suffice for most applications. Experimentation is the only way to see the effect of varying this parameter.', argstr='--numberOfBins %d')
    faceConnected = traits.Bool(desc='This is an advanced parameter. Adjacent voxels are face connected. This affects the connected component algorithm. If this parameter is false, more regions are likely to be identified.', argstr='--faceConnected ')
    minimumObjectSize = traits.Int(desc='Minimum size of object to retain. This parameter can be used to get rid of small regions in noisy images.', argstr='--minimumObjectSize %d')
    inputVolume = File(position=-2, desc='Input volume to be segmented', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')