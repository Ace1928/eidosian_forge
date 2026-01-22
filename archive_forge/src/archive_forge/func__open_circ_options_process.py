from collections.abc import Iterable
import warnings
from typing import Sequence
def _open_circ_options_process(options):
    """For use in both ``_ctrlo_circ`` and ``_target_x``."""
    if options is None:
        options = {}
    new_options = options.copy()
    if 'color' in new_options:
        new_options['facecolor'] = plt.rcParams['axes.facecolor']
        new_options['edgecolor'] = options['color']
        new_options['color'] = None
    else:
        new_options['edgecolor'] = plt.rcParams['lines.color']
        new_options['facecolor'] = plt.rcParams['axes.facecolor']
    if 'linewidth' not in new_options:
        new_options['linewidth'] = plt.rcParams['lines.linewidth']
    if 'zorder' not in new_options:
        new_options['zorder'] = 3
    return new_options