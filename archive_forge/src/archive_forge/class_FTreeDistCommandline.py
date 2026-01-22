from Bio.Application import _Option, _Switch, AbstractCommandline
class FTreeDistCommandline(_EmbossCommandLine):
    """Commandline object for the ftreedist program from EMBOSS.

    ftreedist is an EMBOSS wrapper for the PHYLIP program treedist used for
    calculating distance measures between phylogentic trees.
    """

    def __init__(self, cmd='ftreedist', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-intreefile', 'intreefile'], 'tree file to score (phylip)', filename=True, is_required=True), _Option(['-dtype', 'dtype'], 'distance type ([S]ymetric, [b]ranch score)'), _Option(['-pairing', 'pairing'], 'tree pairing method ([A]djacent pairs, all [p]ossible pairs)'), _Option(['-style', 'style'], 'output style - [V]erbose, [f]ill, [s]parse'), _Option(['-noroot', 'noroot'], 'treat trees as rooted [N/y]'), _Option(['-outgrno', 'outgrno'], 'which taxon to root the trees with (starts from 0)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)