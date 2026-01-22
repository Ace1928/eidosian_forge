import os
from ...base import (
class FindCenterOfBrain(SEMLikeCommandLine):
    """title: Center Of Brain (BRAINS)

    category: Utilities.BRAINS

    description: Finds the center point of a brain

    version: 3.0.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Hans J. Johnson, hans-johnson -at- uiowa.edu, http://wwww.psychiatry.uiowa.edu

    acknowledgements: Hans Johnson(1,3,4); Kent Williams(1);  (1=University of Iowa Department of Psychiatry, 3=University of Iowa Department of Biomedical Engineering, 4=University of Iowa Department of Electrical and Computer Engineering
    """
    input_spec = FindCenterOfBrainInputSpec
    output_spec = FindCenterOfBrainOutputSpec
    _cmd = ' FindCenterOfBrain '
    _outputs_filenames = {'debugClippedImageMask': 'debugClippedImageMask.nii', 'debugTrimmedImage': 'debugTrimmedImage.nii', 'debugDistanceImage': 'debugDistanceImage.nii', 'debugGridImage': 'debugGridImage.nii', 'clippedImageMask': 'clippedImageMask.nii', 'debugAfterGridComputationsForegroundImage': 'debugAfterGridComputationsForegroundImage.nii'}
    _redirect_x = False