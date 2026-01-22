from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def add_dos(self, label: str, dos: Dos) -> None:
    """Adds a dos for plotting.

        Args:
            label: label for the DOS. Must be unique.
            dos: Dos object
        """
    if dos.norm_vol is None:
        self._norm_val = False
    energies = dos.energies - dos.efermi if self.zero_at_efermi else dos.energies
    densities = dos.get_smeared_densities(self.sigma) if self.sigma else dos.densities
    efermi = dos.efermi
    self._doses[label] = {'energies': energies, 'densities': densities, 'efermi': efermi}