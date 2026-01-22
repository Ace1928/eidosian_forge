@_needs_mpl
def _solarized_dark():
    """Apply the solarized light style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams['savefig.facecolor'] = '#002b36'
    plt.rcParams['figure.facecolor'] = '#002b36'
    plt.rcParams['axes.facecolor'] = '#002b36'
    plt.rcParams['patch.edgecolor'] = '#268bd2'
    plt.rcParams['patch.linewidth'] = 3.0
    plt.rcParams['patch.facecolor'] = '#073642'
    plt.rcParams['lines.color'] = '#839496'
    plt.rcParams['text.color'] = '#2aa198'
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['path.sketch'] = None