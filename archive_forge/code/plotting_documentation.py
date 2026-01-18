from __future__ import division
import numpy as np
from pygsp import utils

    Plot the spectrogram of the given graph.

    Parameters
    ----------
    G : Graph
        Graph to analyse.
    node_idx : ndarray
        Order to sort the nodes in the spectrogram

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Ring(15)
    >>> plotting.plot_spectrogram(G)

    