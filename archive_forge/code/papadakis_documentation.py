import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
Design 2 filters with the Papadakis construction (tight frame).

    This function creates a Parseval filter bank of 2 filters.
    The low-pass filter is defined by the function

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\
            \sqrt{1-\frac{\sin\left(\frac{3\pi}{2a}x\right)}{2}} & \mbox{if }a<x\leq \frac{5a}{3} \\
            0 & \mbox{if }x>\frac{5a}{3} \end{cases}

    The high pass filter is adapted to obtain a tight frame.

    Parameters
    ----------
    G : graph
    a : float
        See above equation for this parameter.
        The spectrum is scaled between 0 and 2 (default = 3/4).

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Papadakis(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    