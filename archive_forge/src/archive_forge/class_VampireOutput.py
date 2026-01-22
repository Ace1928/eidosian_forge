from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
class VampireOutput(MSONable):
    """This class processes results from a Vampire Monte Carlo simulation
    and returns the critical temperature.
    """

    def __init__(self, parsed_out=None, nmats=None, critical_temp=None):
        """
        Args:
            parsed_out (json): json rep of parsed stdout DataFrame.
            nmats (int): Number of distinct materials (1 for each specie and up/down spin).
            critical_temp (float): Monte Carlo Tc result.
        """
        self.parsed_out = parsed_out
        self.nmats = nmats
        self.critical_temp = critical_temp