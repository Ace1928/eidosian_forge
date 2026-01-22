import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class PercentileImage(MathsCommand):
    """Use fslmaths to generate a percentile image across a given dimension.

    Examples
    --------
    >>> from nipype.interfaces.fsl.maths import MaxImage
    >>> percer = PercentileImage()
    >>> percer.inputs.in_file = "functional.nii"  # doctest: +SKIP
    >>> percer.dimension = "T"
    >>> percer.perc = 90
    >>> percer.cmdline  # doctest: +SKIP
    'fslmaths functional.nii -Tperc 90 functional_perc.nii'

    """
    input_spec = PercentileImageInput
    _suffix = '_perc'