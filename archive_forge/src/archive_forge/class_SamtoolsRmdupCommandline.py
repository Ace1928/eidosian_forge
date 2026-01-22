from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsRmdupCommandline(AbstractCommandline):
    """Command line wrapper for samtools rmdup.

    Remove potential PCR duplicates, equivalent to::

    $ samtools rmdup [-sS] <input.srt.bam> <out.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsRmdupCommandline
    >>> input_sorted_bam = "/path/to/input.srt.bam"
    >>> out_bam = "/path/to/out.bam"
    >>> rmdup_cmd = SamtoolsRmdupCommandline(input_bam=input_sorted_bam,
    ...                                      out_bam=out_bam)
    >>> print(rmdup_cmd)
    samtools rmdup /path/to/input.srt.bam /path/to/out.bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('rmdup'), _Switch(['-s', 's'], 'Remove duplicates for single-end reads.\n\n                    By default, the command works for paired-end\n                    reads only'), _Switch(['-S', 'S'], 'Treat paired-end reads\n                                    as single-end reads'), _Argument(['in_bam', 'sorted_bam', 'input_bam', 'input', 'input_file'], 'Name Sorted Alignment File ', filename=True, is_required=True), _Argument(['out_bam', 'output_bam', 'output', 'output_file'], 'Output file', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)