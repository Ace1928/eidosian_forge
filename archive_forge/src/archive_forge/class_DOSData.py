from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
class DOSData(metaclass=ABCMeta):
    """Abstract base class for a single series of DOS-like data

    Only the 'info' is a mutable attribute; DOS data is set at init"""

    def __init__(self, info: Info=None) -> None:
        if info is None:
            self.info = {}
        elif isinstance(info, dict):
            self.info = info
        else:
            raise TypeError('Info must be a dict or None')

    @abstractmethod
    def get_energies(self) -> Sequence[float]:
        """Get energy data stored in this object"""

    @abstractmethod
    def get_weights(self) -> Sequence[float]:
        """Get DOS weights stored in this object"""

    @abstractmethod
    def copy(self) -> 'DOSData':
        """Returns a copy in which info dict can be safely mutated"""

    def _sample(self, energies: Sequence[float], width: float=0.1, smearing: str='Gauss') -> np.ndarray:
        """Sample the DOS data at chosen points, with broadening

        Note that no correction is made here for the sampling bin width; total
        intensity will vary with sampling density.

        Args:
            energies: energy values for sampling
            width: Width of broadening kernel
            smearing: selection of broadening kernel (only "Gauss" is currently
                supported)

        Returns:
            Weights sampled from a broadened DOS at values corresponding to x
        """
        self._check_positive_width(width)
        weights_grid = np.zeros(len(energies), float)
        weights = self.get_weights()
        energies = np.asarray(energies, float)
        for i, raw_energy in enumerate(self.get_energies()):
            delta = self._delta(energies, raw_energy, width, smearing=smearing)
            weights_grid += weights[i] * delta
        return weights_grid

    def _almost_equals(self, other: Any) -> bool:
        """Compare with another DOSData for testing purposes"""
        if not isinstance(other, type(self)):
            return False
        if self.info != other.info:
            return False
        if not np.allclose(self.get_weights(), other.get_weights()):
            return False
        return np.allclose(self.get_energies(), other.get_energies())

    @staticmethod
    def _delta(x: np.ndarray, x0: float, width: float, smearing: str='Gauss') -> np.ndarray:
        """Return a delta-function centered at 'x0'.

        This function is used with numpy broadcasting; if x is a row and x0 is
        a column vector, the returned data will be a 2D array with each row
        corresponding to a different delta center.
        """
        if smearing.lower() == 'gauss':
            x1 = -0.5 * ((x - x0) / width) ** 2
            return np.exp(x1) / (np.sqrt(2 * np.pi) * width)
        else:
            msg = 'Requested smearing type not recognized. Got {}'.format(smearing)
            raise ValueError(msg)

    @staticmethod
    def _check_positive_width(width):
        if width <= 0.0:
            msg = 'Cannot add 0 or negative width smearing'
            raise ValueError(msg)

    def sample_grid(self, npts: int, xmin: float=None, xmax: float=None, padding: float=3, width: float=0.1, smearing: str='Gauss') -> 'GridDOSData':
        """Sample the DOS data on an evenly-spaced energy grid

        Args:
            npts: Number of sampled points
            xmin: Minimum sampled x value; if unspecified, a default is chosen
            xmax: Maximum sampled x value; if unspecified, a default is chosen
            padding: If xmin/xmax is unspecified, default value will be padded
                by padding * width to avoid cutting off peaks.
            width: Width of broadening kernel
            smearing: selection of broadening kernel (only 'Gauss' is
                implemented)

        Returns:
            (energy values, sampled DOS)
        """
        if xmin is None:
            xmin = min(self.get_energies()) - padding * width
        if xmax is None:
            xmax = max(self.get_energies()) + padding * width
        energies_grid = np.linspace(xmin, xmax, npts)
        weights_grid = self._sample(energies_grid, width=width, smearing=smearing)
        return GridDOSData(energies_grid, weights_grid, info=self.info.copy())

    def plot(self, npts: int=1000, xmin: float=None, xmax: float=None, width: float=0.1, smearing: str='Gauss', ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
        """Simple 1-D plot of DOS data, resampled onto a grid

        If the special key 'label' is present in self.info, this will be set
        as the label for the plotted line (unless overruled in mplargs). The
        label is only seen if a legend is added to the plot (i.e. by calling
        ``ax.legend()``).

        Args:
            npts, xmin, xmax: output data range, as passed to self.sample_grid
            width: Width of broadening kernel for self.sample_grid()
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
        if mplargs is None:
            mplargs = {}
        if 'label' not in mplargs:
            mplargs.update({'label': self.label_from_info(self.info)})
        return self.sample_grid(npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing).plot(ax=ax, xmin=xmin, xmax=xmax, show=show, filename=filename, mplargs=mplargs)

    @staticmethod
    def label_from_info(info: Dict[str, str]):
        """Generate an automatic legend label from info dict"""
        if 'label' in info:
            return info['label']
        else:
            return '; '.join(map(lambda x: '{}: {}'.format(x[0], x[1]), info.items()))