from Bio.Application import _Option, _Switch, AbstractCommandline
class TranalignCommandline(_EmbossCommandLine):
    """Commandline object for the tranalign program from EMBOSS."""

    def __init__(self, cmd='tranalign', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-asequence', 'asequence'], 'Nucleotide sequences to be aligned.', filename=True, is_required=True), _Option(['-bsequence', 'bsequence'], 'Protein sequence alignment', filename=True, is_required=True), _Option(['-outseq', 'outseq'], 'Output sequence file.', filename=True, is_required=True), _Option(['-table', 'table'], 'Code to use')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)