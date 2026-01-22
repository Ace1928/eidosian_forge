import os
from ...base import (
class BRAINSConstellationDetector(SEMLikeCommandLine):
    """title: Brain Landmark Constellation Detector (BRAINS)

    category: Segmentation.Specialized

    description: This program will find the mid-sagittal plane, a constellation of landmarks in a volume, and create an AC/PC aligned data set with the AC point at the center of the voxel lattice (labeled at the origin of the image physical space.)  Part of this work is an extension of the algorithms originally described by Dr. Babak A. Ardekani, Alvin H. Bachman, Model-based automatic detection of the anterior and posterior commissures on MRI scans, NeuroImage, Volume 46, Issue 3, 1 July 2009, Pages 677-682, ISSN 1053-8119, DOI: 10.1016/j.neuroimage.2009.02.030.  (http://www.sciencedirect.com/science/article/B6WNP-4VRP25C-4/2/8207b962a38aa83c822c6379bc43fe4c)

    version: 1.0

    documentation-url: http://www.nitrc.org/projects/brainscdetector/
    """
    input_spec = BRAINSConstellationDetectorInputSpec
    output_spec = BRAINSConstellationDetectorOutputSpec
    _cmd = ' BRAINSConstellationDetector '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii.gz', 'outputMRML': 'outputMRML.mrml', 'resultsDir': 'resultsDir', 'outputResampledVolume': 'outputResampledVolume.nii.gz', 'outputTransform': 'outputTransform.h5', 'writeBranded2DImage': 'writeBranded2DImage.png', 'outputLandmarksInACPCAlignedSpace': 'outputLandmarksInACPCAlignedSpace.fcsv', 'outputLandmarksInInputSpace': 'outputLandmarksInInputSpace.fcsv', 'outputUntransformedClippedVolume': 'outputUntransformedClippedVolume.nii.gz', 'outputVerificationScript': 'outputVerificationScript.sh'}
    _redirect_x = False