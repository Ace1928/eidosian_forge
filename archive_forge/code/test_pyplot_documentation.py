import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt

        Return a list of all the functions that are mentioned in the
        autosummary blocks contained in *lines*.

        An autosummary block looks like this::

            .. autosummary::
               :toctree: _as_gen
               :template: autosummary.rst
               :nosignatures:

               plot
               plot_date

        