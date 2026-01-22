from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsIdxstatsCommandline(AbstractCommandline):
    """Command line wrapper for samtools idxstats.

    Retrieve and print stats in the index file, equivalent to::

    $ samtools idxstats <aln.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsIdxstatsCommandline
    >>> input = "/path/to/aln_bam"
    >>> samtools_idxstats_cmd = SamtoolsIdxstatsCommandline(input_bam=input)
    >>> print(samtools_idxstats_cmd)
    samtools idxstats /path/to/aln_bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('idxstats'), _Argument(['input', 'in_bam', 'input_bam'], 'BAM file to be indexed')]
        AbstractCommandline.__init__(self, cmd, **kwargs)