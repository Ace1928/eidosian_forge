from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsCalmdCommandline(AbstractCommandline):
    """Command line wrapper for samtools calmd.

    Generate the MD tag, equivalent to::

    $ samtools calmd [-EeubSr] [-C capQcoef] <aln.bam> <ref.fasta>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsCalmdCommandline
    >>> input_bam = "/path/to/aln.bam"
    >>> reference_fasta = "/path/to/reference.fasta"
    >>> calmd_cmd = SamtoolsCalmdCommandline(input_bam=input_bam,
    ...                                      reference=reference_fasta)
    >>> print(calmd_cmd)
    samtools calmd /path/to/aln.bam /path/to/reference.fasta

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('calmd'), _Switch(['-E', 'E'], 'Extended BAQ calculation.\n                    This option trades specificity for sensitivity,\n                    though the effect is minor.'), _Switch(['-e', 'e'], 'Convert the read base to = if it is\n                    identical to the aligned reference base.\n\n                    Indel caller does not support the = bases\n                    at the moment.'), _Switch(['-u', 'u'], 'Output uncompressed BAM'), _Switch(['-b', 'b'], 'Output compressed BAM '), _Switch(['-S', 'S'], 'The input is SAM with header lines '), _Switch(['-r', 'r'], 'Compute the BQ tag (without -A)\n                    or cap base quality by BAQ (with -A).'), _Switch(['-A', 'A'], 'When used jointly with -r this option overwrites\n                    the original base quality'), _Option(['-C', 'C'], 'Coefficient to cap mapping quality\n                    of poorly mapped reads.\n\n                    See the pileup command for details.', equate=False, checker_function=lambda x: isinstance(x, int)), _Argument(['input', 'input_file', 'in_bam', 'infile', 'input_bam'], 'Input BAM', filename=True, is_required=True), _Argument(['reference', 'reference_fasta', 'ref'], 'Reference FASTA to be indexed', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)