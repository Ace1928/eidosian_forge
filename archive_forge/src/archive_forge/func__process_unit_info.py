from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _process_unit_info(self, datasets=None, kwargs=None, *, convert=True):
    """
        Set axis units based on *datasets* and *kwargs*, and optionally apply
        unit conversions to *datasets*.

        Parameters
        ----------
        datasets : list
            List of (axis_name, dataset) pairs (where the axis name is defined
            as in `._axis_map`).  Individual datasets can also be None
            (which gets passed through).
        kwargs : dict
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar) entries) is popped, if present.  Note that this dict is
            mutated in-place!
        convert : bool, default: True
            Whether to return the original datasets or the converted ones.

        Returns
        -------
        list
            Either the original datasets if *convert* is False, or the
            converted ones if *convert* is True (the default).
        """
    datasets = datasets or []
    kwargs = kwargs or {}
    axis_map = self._axis_map
    for axis_name, data in datasets:
        try:
            axis = axis_map[axis_name]
        except KeyError:
            raise ValueError(f'Invalid axis name: {axis_name!r}') from None
        if axis is not None and data is not None and (not axis.have_units()):
            axis.update_units(data)
    for axis_name, axis in axis_map.items():
        if axis is None:
            continue
        units = kwargs.pop(f'{axis_name}units', axis.units)
        if self.name == 'polar':
            polar_units = {'x': 'thetaunits', 'y': 'runits'}
            units = kwargs.pop(polar_units[axis_name], units)
        if units != axis.units and units is not None:
            axis.set_units(units)
            for dataset_axis_name, data in datasets:
                if dataset_axis_name == axis_name and data is not None:
                    axis.update_units(data)
    return [axis_map[axis_name].convert_units(data) if convert and data is not None else data for axis_name, data in datasets]