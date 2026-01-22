import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
class NilearnBaseInterface(LibraryBaseInterface):
    _pkg = 'nilearn'