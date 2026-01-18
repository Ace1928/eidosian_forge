from __future__ import annotations
import collections
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.due import Doi, due

        Args:
            dos: CompleteDos object with project element-orbital DOS.
            Can be obtained from Vasprun.get_complete_dos.
            sigma: Smearing for Gaussian.

        Returns:
            XPS
        