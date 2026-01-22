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
def _make_ticks(self, ax: plt.Axes) -> plt.Axes:
    """Utility private method to add ticks to a band structure."""
    ticks = self.get_ticks()
    uniq_d = []
    uniq_l = []
    temp_ticks = list(zip(ticks['distance'], ticks['label']))
    for idx, t in enumerate(temp_ticks):
        if idx == 0:
            uniq_d.append(t[0])
            uniq_l.append(t[1])
            logger.debug(f'Adding label {t[0]} at {t[1]}')
        elif t[1] == temp_ticks[idx - 1][1]:
            logger.debug(f'Skipping label {t[1]}')
        else:
            logger.debug(f'Adding label {t[0]} at {t[1]}')
            uniq_d.append(t[0])
            uniq_l.append(t[1])
    logger.debug(f'Unique labels are {list(zip(uniq_d, uniq_l))}')
    ax.set_xticks(uniq_d)
    ax.set_xticklabels(uniq_l)
    for idx, label in enumerate(ticks['label']):
        if label is not None:
            if idx != 0:
                if label == ticks['label'][idx - 1]:
                    logger.debug(f'already print label... skipping label {ticks['label'][idx]}')
                else:
                    logger.debug(f'Adding a line at {ticks['distance'][idx]} for label {ticks['label'][idx]}')
                    ax.axvline(ticks['distance'][idx], color='k')
            else:
                logger.debug(f'Adding a line at {ticks['distance'][idx]} for label {ticks['label'][idx]}')
                ax.axvline(ticks['distance'][idx], color='k')
    return ax