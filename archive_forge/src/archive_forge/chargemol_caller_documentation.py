from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
Internal command to process Chargemol CM5 data.

        Args:
            ddec_analysis_path (str): Path VASP_DDEC_analysis.output file

        Returns:
            list[float]: CM5 charges
        