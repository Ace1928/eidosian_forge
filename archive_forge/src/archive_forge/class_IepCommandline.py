from Bio.Application import _Option, _Switch, AbstractCommandline
class IepCommandline(_EmbossCommandLine):
    """Commandline for EMBOSS iep: calculated isoelectric point and charge.

    Examples
    --------
    >>> from Bio.Emboss.Applications import IepCommandline
    >>> iep_cline = IepCommandline(sequence="proteins.faa",
    ...                            outfile="proteins.txt")
    >>> print(iep_cline)
    iep -outfile=proteins.txt -sequence=proteins.faa

    You would typically run the command line with iep_cline() or via the
    Python subprocess module, as described in the Biopython tutorial.

    """

    def __init__(self, cmd='iep', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Protein sequence(s) filename', filename=True, is_required=True), _Option(['-amino', 'amino'], 'Number of N-termini\n\n                    Integer 0 (default) or more.\n                    '), _Option(['-carboxyl', 'carboxyl'], 'Number of C-termini\n\n                    Integer 0 (default) or more.\n                    '), _Option(['-lysinemodified', 'lysinemodified'], 'Number of modified lysines\n\n                    Integer 0 (default) or more.\n                    '), _Option(['-disulphides', 'disulphides'], 'Number of disulphide bridges\n\n                    Integer 0 (default) or more.\n                    '), _Option(['-notermini', 'notermini'], 'Exclude (True) or include (False) charge at N and C terminus.')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)