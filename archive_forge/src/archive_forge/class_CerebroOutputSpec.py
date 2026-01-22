import os
import re as regex
from ..base import (
class CerebroOutputSpec(TraitedSpec):
    outputCerebrumMaskFile = File(desc='path/name of cerebrum mask file')
    outputLabelVolumeFile = File(desc='path/name of label mask file')
    outputAffineTransformFile = File(desc='path/name of affine transform file')
    outputWarpTransformFile = File(desc='path/name of warp transform file')