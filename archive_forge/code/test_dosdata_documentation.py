from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
Check that resampled spectra are independent of the original density

        Compare resampling of sample function on two different grids to the
        same new grid with broadening. We accept a 5% difference because the
        initial shape is slightly different; what we are checking for is a
        factor 2 difference from "double-counting" the extra data points.
        