from nipype.interfaces.base import (
import os
class VBRAINSDemonWarpOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output resampled moving image (will have the same physical space as the fixedVolume).', exists=True)
    outputDisplacementFieldVolume = File(desc='Output deformation field vector image (will have the same physical space as the fixedVolume).', exists=True)
    outputCheckerboardVolume = File(desc='Genete a checkerboard image volume between the fixedVolume and the deformed movingVolume.', exists=True)