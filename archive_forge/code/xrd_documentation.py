from __future__ import annotations
import json
import os
from math import asin, cos, degrees, pi, radians, sin
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.analysis.diffraction.core import (
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        Calculates the diffraction pattern for a structure.

        Args:
            structure (Structure): Input structure
            scaled (bool): Whether to return scaled intensities. The maximum
                peak is set to a value of 100. Defaults to True. Use False if
                you need the absolute values to combine XRD plots.
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.

        Returns:
            DiffractionPattern: XRD pattern
        