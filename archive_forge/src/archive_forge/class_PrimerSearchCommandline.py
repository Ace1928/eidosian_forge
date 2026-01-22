from Bio.Application import _Option, _Switch, AbstractCommandline
class PrimerSearchCommandline(_EmbossCommandLine):
    """Commandline object for the primersearch program from EMBOSS."""

    def __init__(self, cmd='primersearch', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-seqall', '-sequences', 'sequences', 'seqall'], 'Sequence to look for the primer pairs in.', is_required=True), _Option(['-infile', '-primers', 'primers', 'infile'], 'File containing the primer pairs to search for.', filename=True, is_required=True), _Option(['-mismatchpercent', 'mismatchpercent'], 'Allowed percentage mismatch (any integer value, default 0).', is_required=True), _Option(['-snucleotide', 'snucleotide'], 'Sequences are nucleotide (boolean)'), _Option(['-sprotein', 'sprotein'], 'Sequences are protein (boolean)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)