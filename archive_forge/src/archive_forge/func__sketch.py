@_needs_mpl
def _sketch():
    """Apply the sketch style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#D6F5E2'
    plt.rcParams['patch.facecolor'] = '#FFEED4'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['patch.linewidth'] = 3.0
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'black'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['path.sketch'] = (1, 100, 2)