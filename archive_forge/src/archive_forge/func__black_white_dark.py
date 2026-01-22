@_needs_mpl
def _black_white_dark():
    """Apply the black and white dark style to matplotlib's configuration. This functions modifies ``plt.rcParams``."""
    almost_black = '#151515'
    plt.rcParams['savefig.facecolor'] = almost_black
    plt.rcParams['figure.facecolor'] = almost_black
    plt.rcParams['axes.facecolor'] = almost_black
    plt.rcParams['patch.edgecolor'] = 'white'
    plt.rcParams['patch.facecolor'] = almost_black
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['path.sketch'] = None