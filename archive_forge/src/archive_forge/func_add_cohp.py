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
def add_cohp(self, label, cohp) -> None:
    """Adds a COHP for plotting.

        Args:
            label: Label for the COHP. Must be unique.

            cohp: COHP object.
        """
    energies = cohp.energies - cohp.efermi if self.zero_at_efermi else cohp.energies
    populations = cohp.get_cohp()
    int_populations = cohp.get_icohp()
    self._cohps[label] = {'energies': energies, 'COHP': populations, 'ICOHP': int_populations, 'efermi': cohp.efermi}