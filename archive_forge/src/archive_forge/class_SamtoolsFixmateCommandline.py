from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsFixmateCommandline(AbstractCommandline):
    """Command line wrapper for samtools fixmate.

    Fill in mate coordinates, ISIZE and mate related
    flags from a name-sorted alignment, equivalent to::

    $ samtools fixmate <in.nameSrt.bam> <out.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsFixmateCommandline
    >>> in_bam = "/path/to/in.nameSrt.bam"
    >>> out_bam = "/path/to/out.bam"
    >>> fixmate_cmd = SamtoolsFixmateCommandline(input_bam=in_bam,
    ...                                          out_bam=out_bam)
    >>> print(fixmate_cmd)
    samtools fixmate /path/to/in.nameSrt.bam /path/to/out.bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('fixmate'), _Argument(['in_bam', 'sorted_bam', 'input_bam', 'input', 'input_file'], 'Name Sorted Alignment File ', filename=True, is_required=True), _Argument(['out_bam', 'output_bam', 'output', 'output_file'], 'Output file', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)