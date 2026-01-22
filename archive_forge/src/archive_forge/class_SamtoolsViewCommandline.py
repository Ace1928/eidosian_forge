from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsViewCommandline(AbstractCommandline):
    """Command line wrapper for samtools view.

    Extract/print all or sub alignments in SAM or BAM format, equivalent to::

        $ samtools view [-bchuHS] [-t in.refList] [-o output] [-f reqFlag]
                        [-F skipFlag] [-q minMapQ] [-l library] [-r readGroup]
                        [-R rgFile] <in.bam>|<in.sam> [region1 [...]]

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsViewCommandline
    >>> input_file = "/path/to/sam_or_bam_file"
    >>> samtools_view_cmd = SamtoolsViewCommandline(input_file=input_file)
    >>> print(samtools_view_cmd)
    samtools view /path/to/sam_or_bam_file

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('view'), _Switch(['-b', 'b'], 'Output in the BAM format'), _Switch(['-c', 'c'], "Instead of printing the alignments, only count them and\n                    print the total number.\n\n                    All filter options, such as '-f', '-F' and '-q',\n                    are taken into account"), _Switch(['-h', 'h'], 'Include the header in the output'), _Switch(['-u', 'u'], 'Output uncompressed BAM.\n\n                    This option saves time spent on compression/decompression\n                    and is thus preferred when the output is piped to\n                    another samtools command'), _Switch(['-H', 'H'], 'Output the header only'), _Switch(['-S', 'S'], "Input is in SAM.\n                    If @SQ header lines are absent,\n                    the '-t' option is required."), _Option(['-t', 't'], "This file is TAB-delimited.\n                    Each line must contain the reference name and the\n                    length of the reference, one line for each\n                    distinct reference; additional fields are ignored.\n\n                    This file also defines the order of the reference\n                    sequences in sorting.\n                    If you run   'samtools faidx <ref.fa>',\n                    the resultant index file <ref.fa>.fai can be used\n                    as this <in.ref_list> file.", filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-o', 'o'], 'Output file', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-f', 'f'], 'Only output alignments with all bits in\n                    INT present in the FLAG field', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-F', 'F'], 'Skip alignments with bits present in INT', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-q', 'q'], 'Skip alignments with MAPQ smaller than INT', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-r', 'r'], 'Only output reads in read group STR', equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-R', 'R'], 'Output reads in read groups listed in FILE', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-l', 'l'], 'Only output reads in library STR', equate=False, checker_function=lambda x: isinstance(x, str)), _Switch(['-1', 'fast_bam'], 'Use zlib compression level 1 to compress the output'), _Argument(['input', 'input_file'], 'Input File Name', filename=True, is_required=True), _Argument(['region'], 'Region', is_required=False)]
        AbstractCommandline.__init__(self, cmd, **kwargs)