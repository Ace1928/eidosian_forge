from Bio.Application import _Option, _Switch, AbstractCommandline
class WaterCommandline(_EmbossCommandLine):
    """Commandline object for the water program from EMBOSS."""

    def __init__(self, cmd='water', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-asequence', 'asequence'], 'First sequence to align', filename=True, is_required=True), _Option(['-bsequence', 'bsequence'], 'Second sequence to align', filename=True, is_required=True), _Option(['-gapopen', 'gapopen'], 'Gap open penalty', is_required=True), _Option(['-gapextend', 'gapextend'], 'Gap extension penalty', is_required=True), _Option(['-datafile', 'datafile'], 'Matrix file', filename=True), _Switch(['-nobrief', 'nobrief'], 'Display extended identity and similarity'), _Switch(['-brief', 'brief'], 'Display brief identity and similarity'), _Option(['-similarity', 'similarity'], 'Display percent identity and similarity'), _Option(['-snucleotide', 'snucleotide'], 'Sequences are nucleotide (boolean)'), _Option(['-sprotein', 'sprotein'], 'Sequences are protein (boolean)'), _Option(['-aformat', 'aformat'], 'Display output in a different specified output format')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)