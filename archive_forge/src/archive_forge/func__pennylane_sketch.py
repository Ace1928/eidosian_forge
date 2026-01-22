@_needs_mpl
def _pennylane_sketch():
    """Apply the PennyLane-Sketch style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    _pennylane()
    plt.rcParams['path.sketch'] = (1, 250, 1)