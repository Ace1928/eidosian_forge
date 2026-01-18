import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
Plots the real and imaginary parts of the density matrix as two 3D bar plots.

        Args:
            axes: A list of 2 `plt.Axes` instances. Note that they must be in
                3d projections. If not given, a new figure is created with 2
                axes and the plotted figure is shown.
            **plot_kwargs: The optional kwargs passed to bar3d.

        Returns:
            the list of `plt.Axes` being plotted on.

        Raises:
            ValueError: If axes is a list with length != 2.
        