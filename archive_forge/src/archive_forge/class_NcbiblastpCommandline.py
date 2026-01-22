from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbiblastpCommandline(_NcbiblastMain2SeqCommandline):
    """Create a commandline for the NCBI BLAST+ program blastp (for proteins).

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old blastall tool with separate tools for each of the searches.
    This wrapper therefore replaces BlastallCommandline with option -p blastp.

    >>> from Bio.Blast.Applications import NcbiblastpCommandline
    >>> cline = NcbiblastpCommandline(query="rosemary.pro", db="nr",
    ...                               evalue=0.001, remote=True, ungapped=True)
    >>> cline
    NcbiblastpCommandline(cmd='blastp', query='rosemary.pro', db='nr', evalue=0.001, remote=True, ungapped=True)
    >>> print(cline)
    blastp -query rosemary.pro -db nr -evalue 0.001 -remote -ungapped

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='blastp', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-task', 'task'], 'Task to execute (string, blastp (default), blastp-fast or blastp-short).', checker_function=lambda value: value in ['blastp', 'blastp-fast', 'blastp-short'], equate=False), _Option(['-matrix', 'matrix'], 'Scoring matrix name (default BLOSUM62).'), _Option(['-threshold', 'threshold'], 'Minimum score for words to be added to the BLAST lookup table (float).', equate=False), _Option(['-comp_based_stats', 'comp_based_stats'], 'Use composition-based statistics (string, default 2, i.e. True).\n\n0, F or f: no composition-based statistics\n\n2, T or t, D or d : Composition-based score adjustment as in Bioinformatics 21:902-911, 2005, conditioned on sequence properties\n\nNote that tblastn also supports values of 1 and 3.', checker_function=lambda value: value in '0Ft2TtDd', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable\nDefault is "12 2.2 2.5"', equate=False), _Switch(['-ungapped', 'ungapped'], 'Perform ungapped alignment only?'), _Switch(['-use_sw_tback', 'use_sw_tback'], 'Compute locally optimal Smith-Waterman alignments?')]
        _NcbiblastMain2SeqCommandline.__init__(self, cmd, **kwargs)