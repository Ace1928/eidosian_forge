@_needs_mpl
def _pennylane():
    """Apply the PennyLane style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    almost_black = '#151515'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#FFB5F1'
    plt.rcParams['patch.facecolor'] = '#D5F0FD'
    plt.rcParams['patch.edgecolor'] = almost_black
    plt.rcParams['patch.linewidth'] = 2.0
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'black'
    plt.rcParams['text.color'] = 'black'
    if 'Quicksand' in {font.name for font in fm.FontManager().ttflist}:
        plt.rcParams['font.family'] = 'Quicksand'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['path.sketch'] = None