from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsMpileupCommandline(AbstractCommandline):
    """Command line wrapper for samtools mpileup.

    Generate BCF or pileup for one or multiple BAM files, equivalent to::

        $ samtools mpileup [-EBug] [-C capQcoef] [-r reg] [-f in.fa]
                           [-l list] [-M capMapQ] [-Q minBaseQ]
                           [-q minMapQ] in.bam [in2.bam [...]]

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsMpileupCommandline
    >>> input = ["/path/to/sam_or_bam_file"]
    >>> samtools_mpileup_cmd = SamtoolsMpileupCommandline(input_file=input)
    >>> print(samtools_mpileup_cmd)
    samtools mpileup /path/to/sam_or_bam_file

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('mpileup'), _Switch(['-E', 'E'], 'Extended BAQ computation.\n                    This option helps sensitivity especially\n                    for MNPs, but may hurt specificity a little bit'), _Switch(['-B', 'B'], 'Disable probabilistic realignment for the\n                    computation of base alignment quality (BAQ).\n\n                    BAQ is the Phred-scaled probability of a read base being\n                    misaligned.\n                    Applying this option greatly helps to reduce false SNPs\n                    caused by misalignments'), _Switch(['-g', 'g'], 'Compute genotype likelihoods and output them in the\n                    binary call format (BCF)'), _Switch(['-u', 'u'], 'Similar to -g except that the output is\n                    uncompressed BCF, which is preferred for piping'), _Option(['-C', 'C'], 'Coefficient for downgrading mapping quality for\n                    reads containing excessive mismatches.\n\n                    Given a read with a phred-scaled probability q of\n                    being generated from the mapped position,\n                    the new mapping quality is about sqrt((INT-q)/INT)*INT.\n                    A zero value disables this functionality;\n                    if enabled, the recommended value for BWA is 50', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-r', 'r'], 'Only generate pileup in region STR', equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-f', 'f'], 'The faidx-indexed reference file in the FASTA format.\n\n                    The file can be optionally compressed by razip', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-l', 'l'], 'BED or position list file containing a list of regions\n                    or sites where pileup or BCF should be generated', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-M', 'M'], 'Cap Mapping Quality at M', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-q', 'q'], 'Minimum mapping quality for an alignment to be used', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-Q', 'Q'], 'Minimum base quality for a base to be considered', equate=False, checker_function=lambda x: isinstance(x, int)), _Switch(['-6', 'illumina_13'], 'Assume the quality is in the Illumina 1.3+ encoding'), _Switch(['-A', 'A'], 'Do not skip anomalous read pairs in variant calling.'), _Option(['-b', 'b'], 'List of input BAM files, one file per line', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-d', 'd'], 'At a position, read maximally INT reads per input BAM', equate=False, checker_function=lambda x: isinstance(x, int)), _Switch(['-D', 'D'], 'Output per-sample read depth'), _Switch(['-S', 'S'], 'Output per-sample Phred-scaled\n                                strand bias P-value'), _Option(['-e', 'e'], 'Phred-scaled gap extension sequencing error probability.\n\n                    Reducing INT leads to longer indels', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-h', 'h'], 'Coefficient for modeling homopolymer errors.\n\n                    Given an l-long homopolymer run, the sequencing error\n                    of an indel of size s is modeled as INT*s/l', equate=False, checker_function=lambda x: isinstance(x, int)), _Switch(['-I', 'I'], 'Do not perform INDEL calling'), _Option(['-L', 'L'], 'Skip INDEL calling if the average per-sample\n                    depth is above INT', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-o', 'o'], 'Phred-scaled gap open sequencing error probability.\n\n                    Reducing INT leads to more indel calls.', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-p', 'p'], 'Comma delimited list of platforms (determined by @RG-PL)\n                    from which indel candidates are obtained.\n\n                    It is recommended to collect indel candidates from\n                    sequencing technologies that have low indel error rate\n                    such as ILLUMINA', equate=False, checker_function=lambda x: isinstance(x, str)), _ArgumentList(['input_file'], 'Input File for generating mpileup', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)