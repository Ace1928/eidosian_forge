from nipype.interfaces.base import (
import os
class BRAINSFitOutputSpec(TraitedSpec):
    bsplineTransform = File(desc='(optional) Filename to which save the estimated transform. NOTE: You must set at least one output object (either a deformed image or a transform.  NOTE: USE THIS ONLY IF THE FINAL TRANSFORM IS BSpline', exists=True)
    linearTransform = File(desc='(optional) Filename to which save the estimated transform. NOTE: You must set at least one output object (either a deformed image or a transform.  NOTE: USE THIS ONLY IF THE FINAL TRANSFORM IS ---NOT--- BSpline', exists=True)
    outputVolume = File(desc='(optional) Output image for registration. NOTE: You must select either the outputTransform or the outputVolume option.', exists=True)
    outputFixedVolumeROI = File(desc='The ROI automatically found in fixed image, ONLY FOR ROIAUTO mode.', exists=True)
    outputMovingVolumeROI = File(desc='The ROI automatically found in moving image, ONLY FOR ROIAUTO mode.', exists=True)
    strippedOutputTransform = File(desc='File name for the rigid component of the estimated affine transform. Can be used to rigidly register the moving image to the fixed image. NOTE:  This value is overwritten if either bsplineTransform or linearTransform is set.', exists=True)
    outputTransform = File(desc='(optional) Filename to which save the (optional) estimated transform. NOTE: You must select either the outputTransform or the outputVolume option.', exists=True)