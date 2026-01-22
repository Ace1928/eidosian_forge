from Bio.Application import _Option, _Switch, AbstractCommandline
class FDNADistCommandline(_EmbossCommandLine):
    """Commandline object for the fdnadist program from EMBOSS.

    fdnadist is an EMBOSS wrapper for the PHYLIP program dnadist for
    calculating distance matrices from DNA sequence files.
    """

    def __init__(self, cmd='fdnadist', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'seq file to use (phylip)', filename=True, is_required=True), _Option(['-method', 'method'], 'sub. model [f,k,j,l,s]', is_required=True), _Option(['-gamma', 'gamma'], 'gamma [g, i,n]'), _Option(['-ncategories', 'ncategories'], 'number of rate catergories (1-9)'), _Option(['-rate', 'rate'], 'rate for each category'), _Option(['-categories', 'categories'], 'File of substitution rate categories'), _Option(['-weights', 'weights'], 'weights file'), _Option(['-gammacoefficient', 'gammacoefficient'], 'value for gamma (> 0.001)'), _Option(['-invarfrac', 'invarfrac'], 'proportoin of invariant sites'), _Option(['-ttratio', 'ttratio'], 'ts/tv ratio'), _Option(['-freqsfrom', 'freqsfrom'], 'use emprical base freqs'), _Option(['-basefreq', 'basefreq'], 'specify basefreqs'), _Option(['-lower', 'lower'], 'lower triangle matrix (y/N)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)