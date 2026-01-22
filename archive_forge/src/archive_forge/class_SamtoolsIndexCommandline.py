from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsIndexCommandline(AbstractCommandline):
    """Command line wrapper for samtools index.

    Index sorted alignment for fast random access, equivalent to::

    $ samtools index <aln.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsIndexCommandline
    >>> input = "/path/to/aln_bam"
    >>> samtools_index_cmd = SamtoolsIndexCommandline(input_bam=input)
    >>> print(samtools_index_cmd)
    samtools index /path/to/aln_bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('index'), _Argument(['input', 'in_bam', 'input_bam'], 'BAM file to be indexed')]
        AbstractCommandline.__init__(self, cmd, **kwargs)