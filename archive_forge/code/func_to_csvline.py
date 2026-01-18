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
def to_csvline(self, with_header=False):
    """Return a string with data in CSV format. Add header if `with_header`."""
    string = ''
    if with_header:
        string += f'# {' '.join((at for at in AbinitTimerSection.FIELDS))}\n'
    string += ', '.join((str(v) for v in self.to_tuple())) + '\n'
    return string