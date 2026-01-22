import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
class ImageFileSPM(ImageFile):
    """Defines a trait whose value must be a NIfTI file."""

    def __init__(self, value=NoDefaultSpecified, exists=False, resolve=False, **metadata):
        """Create an ImageFileSPM trait."""
        super(ImageFileSPM, self).__init__(value=value, exists=exists, types=['nifti1', 'nifti2'], allow_compressed=False, resolve=resolve, **metadata)