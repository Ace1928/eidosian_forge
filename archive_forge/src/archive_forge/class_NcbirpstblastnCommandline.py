from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbirpstblastnCommandline(_NcbiblastCommandline):
    """Wrapper for the NCBI BLAST+ program rpstblastn.

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old rpsblast tool with a similar tool of the same name, and a
    separate tool rpstblastn for Translated Reverse Position Specific BLAST.

    >>> from Bio.Blast.Applications import NcbirpstblastnCommandline
    >>> cline = NcbirpstblastnCommandline(help=True)
    >>> cline
    NcbirpstblastnCommandline(cmd='rpstblastn', help=True)
    >>> print(cline)
    rpstblastn -help

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='rpstblastn', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-strand', 'strand'], 'Query strand(s) to search against database/subject.\n\nValues allowed are "both" (default), "minus", "plus".', checker_function=lambda value: value in ['both', 'minus', 'plus'], equate=False), _Option(['-query_gencode', 'query_gencode'], 'Genetic code to use to translate query (integer, default 1).', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable. Default is "12 2.2 2.5"', equate=False), _Option(['-comp_based_stats', 'comp_based_stats'], 'Use composition-based statistics.\n\nD or d: default (equivalent to 0)\n\n0 or F or f: Simplified Composition-based statistics as in Bioinformatics 15:1000-1011, 1999\n\n1 or T or t: Composition-based statistics as in NAR 29:2994-3005, 2001\n\nDefault = 0.', checker_function=lambda value: value in 'Dd0Ff1Tt', equate=False), _Switch(['-ungapped', 'ungapped'], 'Perform ungapped alignment only?'), _Switch(['-use_sw_tback', 'use_sw_tback'], 'Compute locally optimal Smith-Waterman alignments?')]
        _NcbiblastCommandline.__init__(self, cmd, **kwargs)