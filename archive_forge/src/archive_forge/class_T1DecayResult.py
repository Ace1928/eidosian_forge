from typing import Any, Optional, TYPE_CHECKING
import warnings
import pandas as pd
import sympy
from matplotlib import pyplot as plt
import numpy as np
from cirq import circuits, ops, study, value, _import
from cirq._compat import proper_repr
class T1DecayResult:
    """Results from a Rabi oscillation experiment."""

    def __init__(self, data: pd.DataFrame):
        """Inits T1DecayResult.

        Args:
            data: A data frame with three columns:
                delay_ns, false_count, true_count.
        """
        assert list(data.columns) == ['delay_ns', 'false_count', 'true_count']
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """A data frame with delay_ns, false_count, true_count columns."""
        return self._data

    @property
    def constant(self) -> float:
        """The t1 decay constant."""

        def exp_decay(x, t1):
            return np.exp(-x / t1)
        xs = self._data['delay_ns']
        ts = self._data['true_count']
        fs = self._data['false_count']
        probs = ts / (fs + ts)
        guess_index = np.argmin(np.abs(probs - 1.0 / np.e))
        t1_guess = xs[guess_index]
        try:
            popt, _ = optimize.curve_fit(exp_decay, xs, probs, p0=[t1_guess])
            t1 = popt[0]
            return t1
        except RuntimeError:
            warnings.warn('Optimal parameters could not be found for curve fit', RuntimeWarning)
            return np.nan

    def plot(self, ax: Optional[plt.Axes]=None, include_fit: bool=False, **plot_kwargs: Any) -> plt.Axes:
        """Plots the excited state probability vs the amount of delay.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            include_fit: boolean to include exponential decay fit on graph
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=0, ymax=1)
        xs = self._data['delay_ns']
        ts = self._data['true_count']
        fs = self._data['false_count']
        ax.plot(xs, ts / (fs + ts), 'ro-', **plot_kwargs)
        if include_fit and (not np.isnan(self.constant)):
            ax.plot(xs, np.exp(-xs / self.constant), label='curve fit')
            plt.legend()
        ax.set_xlabel('Delay between initialization and measurement (nanoseconds)')
        ax.set_ylabel('Excited State Probability')
        ax.set_title('T1 Decay Experiment Data')
        if show_plot:
            fig.show()
        return ax

    def __str__(self) -> str:
        return f'T1DecayResult with data:\n{self.data}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data.equals(other.data)

    def __ne__(self, other) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return f'cirq.experiments.T1DecayResult(data={proper_repr(self.data)})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            p.text('T1DecayResult(...)')
        else:
            p.text(str(self))