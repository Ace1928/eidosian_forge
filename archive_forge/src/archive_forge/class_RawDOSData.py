from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
class RawDOSData(GeneralDOSData):
    """A collection of weighted delta functions which sum to form a DOS

    This is an appropriate data container for density-of-states (DOS) or
    spectral data where the energy data values not form a known regular
    grid. The data may be plotted or resampled for further analysis using the
    sample_grid() and plot() methods. Multiple weights at the same
    energy value will *only* be combined in output data, and data stored in
    RawDOSData is never resampled. A plot_deltas() function is also provided
    which plots the raw data.

    Metadata may be stored in the info dict, in which keys and values must be
    strings. This data is used for selecting and combining multiple DOSData
    objects in a DOSCollection object.

    When RawDOSData objects are combined with the addition operator::

      big_dos = raw_dos_1 + raw_dos_2

    the energy and weights data is *concatenated* (i.e. combined without
    sorting or replacement) and the new info dictionary consists of the
    *intersection* of the inputs: only key-value pairs that were common to both
    of the input objects will be retained in the new combined object. For
    example::

      (RawDOSData([x1], [y1], info={'symbol': 'O', 'index': '1'})
       + RawDOSData([x2], [y2], info={'symbol': 'O', 'index': '2'}))

    will yield the equivalent of::

      RawDOSData([x1, x2], [y1, y2], info={'symbol': 'O'})

    """

    def __add__(self, other: 'RawDOSData') -> 'RawDOSData':
        if not isinstance(other, RawDOSData):
            raise TypeError('RawDOSData can only be combined with other RawDOSData objects')
        new_info = dict(set(self.info.items()) & set(other.info.items()))
        new_data = np.concatenate((self._data, other._data), axis=1)
        new_object = RawDOSData([], [], info=new_info)
        new_object._data = new_data
        return new_object

    def plot_deltas(self, ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
        """Simple plot of sparse DOS data as a set of delta functions

        Items at the same x-value can overlap and will not be summed together

        Args:
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib Axes.vlines
                command (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        """
        if mplargs is None:
            mplargs = {}
        with SimplePlottingAxes(ax=ax, show=show, filename=filename) as ax:
            ax.vlines(self.get_energies(), 0, self.get_weights(), **mplargs)
        return ax