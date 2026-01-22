from Bio.Application import _Option, _Switch, AbstractCommandline
class SeqretCommandline(_EmbossMinimalCommandLine):
    """Commandline object for the seqret program from EMBOSS.

    This tool allows you to interconvert between different sequence file
    formats (e.g. GenBank to FASTA). Combining Biopython's Bio.SeqIO module
    with seqret using a suitable intermediate file format can allow you to
    read/write to an even wider range of file formats.

    This wrapper currently only supports the core functionality, things like
    feature tables (in EMBOSS 6.1.0 onwards) are not yet included.
    """

    def __init__(self, cmd='seqret', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Input sequence(s) filename', filename=True), _Option(['-outseq', 'outseq'], 'Output sequence file.', filename=True), _Option(['-sformat', 'sformat'], 'Input sequence(s) format (e.g. fasta, genbank)'), _Option(['-osformat', 'osformat'], 'Output sequence(s) format (e.g. fasta, genbank)')]
        _EmbossMinimalCommandLine.__init__(self, cmd, **kwargs)

    def _validate(self):
        if not (self.outseq or self.filter or self.stdout):
            raise ValueError('You must either set outfile (output filename), or enable filter or stdout (output to stdout).')
        if not (self.sequence or self.filter or self.stdint):
            raise ValueError('You must either set sequence (input filename), or enable filter or stdin (input from stdin).')
        return _EmbossMinimalCommandLine._validate(self)