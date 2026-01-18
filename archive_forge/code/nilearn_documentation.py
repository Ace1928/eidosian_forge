import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
takes a 3-dimensional numpy array and an affine,
        returns the equivalent 4th dimensional nifti file