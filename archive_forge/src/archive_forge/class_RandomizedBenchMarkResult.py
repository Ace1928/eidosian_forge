import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
class RandomizedBenchMarkResult:
    """Results from a randomized benchmarking experiment."""

    def __init__(self, num_cliffords: Sequence[int], ground_state_probabilities: Sequence[float]):
        """Inits RandomizedBenchMarkResult.

        Args:
            num_cliffords: The different numbers of Cliffords in the RB
                study.
            ground_state_probabilities: The corresponding average ground state
                probabilities.
        """
        self._num_cfds_seq = num_cliffords
        self._gnd_state_probs = ground_state_probabilities

    @property
    def data(self) -> Sequence[Tuple[int, float]]:
        """Returns a sequence of tuple pairs with the first item being a
        number of Cliffords and the second item being the corresponding average
        ground state probability.
        """
        return [(num, prob) for num, prob in zip(self._num_cfds_seq, self._gnd_state_probs)]

    def plot(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the average ground state probability vs the number of
        Cliffords in the RB study.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax = cast(plt.Axes, ax)
        ax.set_ylim((0.0, 1.0))
        ax.plot(self._num_cfds_seq, self._gnd_state_probs, 'ro-', **plot_kwargs)
        ax.set_xlabel('Number of Cliffords')
        ax.set_ylabel('Ground State Probability')
        if show_plot:
            fig.show()
        return ax