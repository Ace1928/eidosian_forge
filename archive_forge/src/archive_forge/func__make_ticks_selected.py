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
def _make_ticks_selected(self, ax: plt.Axes, branches: list[int]) -> tuple[plt.Axes, list[float]]:
    """Utility private method to add ticks to a band structure with selected branches."""
    if not ax.figure:
        fig = plt.figure()
        ax.set_figure(fig)
    ticks = self.get_ticks()
    distance = []
    label = []
    rm_elems = []
    for idx in range(1, len(ticks['distance'])):
        if ticks['label'][idx] == ticks['label'][idx - 1]:
            rm_elems.append(idx)
    for idx in range(len(ticks['distance'])):
        if idx not in rm_elems:
            distance.append(ticks['distance'][idx])
            label.append(ticks['label'][idx])
    l_branches = [distance[i] - distance[i - 1] for i in range(1, len(distance))]
    n_distance = []
    n_label = []
    for branch in branches:
        n_distance.append(l_branches[branch])
        if '$\\mid$' not in label[branch] and '$\\mid$' not in label[branch + 1]:
            n_label.append([label[branch], label[branch + 1]])
        elif '$\\mid$' in label[branch] and '$\\mid$' not in label[branch + 1]:
            n_label.append([label[branch].split('$')[-1], label[branch + 1]])
        elif '$\\mid$' not in label[branch] and '$\\mid$' in label[branch + 1]:
            n_label.append([label[branch], label[branch + 1].split('$')[0]])
        else:
            n_label.append([label[branch].split('$')[-1], label[branch + 1].split('$')[0]])
    f_distance: list[float] = []
    rf_distance: list[float] = []
    f_label: list[str] = []
    f_label.extend((n_label[0][0], n_label[0][1]))
    f_distance.extend((0.0, n_distance[0]))
    rf_distance.extend((0.0, n_distance[0]))
    length = n_distance[0]
    for idx in range(1, len(n_distance)):
        if n_label[idx][0] == n_label[idx - 1][1]:
            f_distance.extend((length, length + n_distance[idx]))
            f_label.extend((n_label[idx][0], n_label[idx][1]))
        else:
            f_distance.append(length + n_distance[idx])
            f_label[-1] = n_label[idx - 1][1] + '$\\mid$' + n_label[idx][0]
            f_label.append(n_label[idx][1])
        rf_distance.append(length + n_distance[idx])
        length += n_distance[idx]
    uniq_d = []
    uniq_l = []
    temp_ticks = list(zip(f_distance, f_label))
    for idx, tick in enumerate(temp_ticks):
        if idx == 0:
            uniq_d.append(tick[0])
            uniq_l.append(tick[1])
            logger.debug(f'Adding label {tick[0]} at {tick[1]}')
        elif tick[1] == temp_ticks[idx - 1][1]:
            logger.debug(f'Skipping label {tick[1]}')
        else:
            logger.debug(f'Adding label {tick[0]} at {tick[1]}')
            uniq_d.append(tick[0])
            uniq_l.append(tick[1])
    logger.debug(f'Unique labels are {list(zip(uniq_d, uniq_l))}')
    ax.set_xticks(uniq_d)
    ax.set_xticklabels(uniq_l)
    for idx in range(len(f_label)):
        if f_label[idx] is not None:
            if idx != 0:
                if f_label[idx] == f_label[idx - 1]:
                    logger.debug(f'already print label... skipping label {f_label[idx]}')
                else:
                    logger.debug(f'Adding a line at {f_distance[idx]} for label {f_label[idx]}')
                    ax.axvline(f_distance[idx], color='k')
            else:
                logger.debug(f'Adding a line at {f_distance[idx]} for label {f_label[idx]}')
                ax.axvline(f_distance[idx], color='k')
    shift = []
    br = -1
    for branch in branches:
        br += 1
        shift.append(distance[branch] - rf_distance[br])
    return (ax, shift)