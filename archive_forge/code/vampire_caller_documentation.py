from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

        Args:
            parsed_out (json): json rep of parsed stdout DataFrame.
            nmats (int): Number of distinct materials (1 for each specie and up/down spin).
            critical_temp (float): Monte Carlo Tc result.
        