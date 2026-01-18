from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def timers(self, filename=None, mpi_rank='0'):
    """Return the list of timers associated to the given `filename` and MPI rank mpi_rank."""
    if filename is not None:
        return [self._timers[filename][mpi_rank]]
    return [self._timers[filename][mpi_rank] for filename in self._filenames]