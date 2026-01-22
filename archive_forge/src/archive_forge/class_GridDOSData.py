from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
class GridDOSData(GeneralDOSData):
    """A collection of regularly-sampled data which represents a DOS

    This is an appropriate data container for density-of-states (DOS) or
    spectral data where the intensity values form a regular grid. This
    is generally the result of sampling or integrating into discrete
    bins, rather than a collection of unique states. The data may be
    plotted or resampled for further analysis using the sample_grid()
    and plot() methods.

    Metadata may be stored in the info dict, in which keys and values must be
    strings. This data is used for selecting and combining multiple DOSData
    objects in a DOSCollection object.

    When RawDOSData objects are combined with the addition operator::

      big_dos = raw_dos_1 + raw_dos_2

    the weights data is *summed* (requiring a consistent energy grid) and the
    new info dictionary consists of the *intersection* of the inputs: only
    key-value pairs that were common to both of the input objects will be
    retained in the new combined object. For example::

      (GridDOSData([0.1, 0.2, 0.3], [y1, y2, y3],
                   info={'symbol': 'O', 'index': '1'})
       + GridDOSData([0.1, 0.2, 0.3], [y4, y5, y6],
                     info={'symbol': 'O', 'index': '2'}))

    will yield the equivalent of::

      GridDOSData([0.1, 0.2, 0.3], [y1+y4, y2+y5, y3+y6], info={'symbol': 'O'})

    """

    def __init__(self, energies: Sequence[float], weights: Sequence[float], info: Info=None) -> None:
        n_entries = len(energies)
        if not np.allclose(energies, np.linspace(energies[0], energies[-1], n_entries)):
            raise ValueError('Energies must be an evenly-spaced 1-D grid')
        if len(weights) != n_entries:
            raise ValueError('Energies and weights must be the same length')
        super().__init__(energies, weights, info=info)
        self.sigma_cutoff = 3

    def _check_spacing(self, width) -> float:
        current_spacing = self._data[0, 1] - self._data[0, 0]
        if width < 2 * current_spacing:
            warnings.warn('The broadening width is small compared to the original sampling density. The results are unlikely to be smooth.')
        return current_spacing

    def _sample(self, energies: Sequence[float], width: float=0.1, smearing: str='Gauss') -> np.ndarray:
        current_spacing = self._check_spacing(width)
        return super()._sample(energies=energies, width=width, smearing=smearing) * current_spacing

    def __add__(self, other: 'GridDOSData') -> 'GridDOSData':
        if not isinstance(other, GridDOSData):
            raise TypeError('GridDOSData can only be combined with other GridDOSData objects')
        if len(self._data[0, :]) != len(other.get_energies()):
            raise ValueError('Cannot add GridDOSData objects with different-length energy grids.')
        if not np.allclose(self._data[0, :], other.get_energies()):
            raise ValueError('Cannot add GridDOSData objects with different energy grids.')
        new_info = dict(set(self.info.items()) & set(other.info.items()))
        new_weights = self._data[1, :] + other.get_weights()
        new_object = GridDOSData(self._data[0, :], new_weights, info=new_info)
        return new_object

    @staticmethod
    def _interpret_smearing_args(npts: int, width: float=None, default_npts: int=1000, default_width: float=0.1) -> Tuple[int, Union[float, None]]:
        """Figure out what the user intended: resample if width provided"""
        if width is not None:
            if npts:
                return (npts, float(width))
            else:
                return (default_npts, float(width))
        elif npts:
            return (npts, default_width)
        else:
            return (0, None)

    def plot(self, npts: int=0, xmin: float=None, xmax: float=None, width: float=None, smearing: str='Gauss', ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
        """Simple 1-D plot of DOS data

        Data will be resampled onto a grid with `npts` points unless `npts` is
        set to zero, in which case:

        - no resampling takes place
        - `width` and `smearing` are ignored
        - `xmin` and `xmax` affect the axis limits of the plot, not the
          underlying data.

        If the special key 'label' is present in self.info, this will be set
        as the label for the plotted line (unless overruled in mplargs). The
        label is only seen if a legend is added to the plot (i.e. by calling
        ``ax.legend()``).

        Args:
            npts, xmin, xmax: output data range, as passed to self.sample_grid
            width: Width of broadening kernel, passed to self.sample_grid().
                If no npts was set but width is set, npts will be set to 1000.
            smearing: selection of broadening kernel for self.sample_grid()
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib plot command
                (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        """
        npts, width = self._interpret_smearing_args(npts, width)
        if mplargs is None:
            mplargs = {}
        if 'label' not in mplargs:
            mplargs.update({'label': self.label_from_info(self.info)})
        if npts:
            assert isinstance(width, float)
            dos = self.sample_grid(npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing)
        else:
            dos = self
        energies, intensity = (dos.get_energies(), dos.get_weights())
        with SimplePlottingAxes(ax=ax, show=show, filename=filename) as ax:
            ax.plot(energies, intensity, **mplargs)
            ax.set_xlim(left=xmin, right=xmax)
        return ax