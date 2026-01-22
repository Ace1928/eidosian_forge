from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbipsiblastCommandline(_Ncbiblast2SeqCommandline):
    """Wrapper for the NCBI BLAST+ program psiblast.

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old blastpgp tool with a similar tool psiblast. This wrapper
    therefore replaces BlastpgpCommandline, the wrapper for blastpgp.

    >>> from Bio.Blast.Applications import NcbipsiblastCommandline
    >>> cline = NcbipsiblastCommandline(help=True)
    >>> cline
    NcbipsiblastCommandline(cmd='psiblast', help=True)
    >>> print(cline)
    psiblast -help

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='psiblast', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-matrix', 'matrix'], 'Scoring matrix name (default BLOSUM62).', equate=False), _Option(['-threshold', 'threshold'], 'Minimum score for words to be added to the BLAST lookup table (float).', equate=False), _Option(['-comp_based_stats', 'comp_based_stats'], 'Use composition-based statistics (string, default 2, i.e. True).\n\n0, F or f: no composition-based statistics\n\n2, T or t, D or d : Composition-based score adjustment as in Bioinformatics 21:902-911, 2005, conditioned on sequence properties\n\nNote that tblastn also supports values of 1 and 3.', checker_function=lambda value: value in '0Ft2TtDd', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable. Default is "12 2.2 2.5"', equate=False), _Option(['-gap_trigger', 'gap_trigger'], 'Number of bits to trigger gapping (float, default 22).', equate=False), _Switch(['-use_sw_tback', 'use_sw_tback'], 'Compute locally optimal Smith-Waterman alignments?'), _Option(['-num_iterations', 'num_iterations'], 'Number of iterations to perform (integer, at least one).\n\nDefault is one. Incompatible with: remote', equate=False), _Option(['-out_pssm', 'out_pssm'], 'File name to store checkpoint file.', filename=True, equate=False), _Option(['-out_ascii_pssm', 'out_ascii_pssm'], 'File name to store ASCII version of PSSM.', filename=True, equate=False), _Switch(['-save_pssm_after_last_round', 'save_pssm_after_last_round'], 'Save PSSM after the last database search.'), _Switch(['-save_each_pssm', 'save_each_pssm'], 'Save PSSM after each iteration\n\nFile name is given in -save_pssm or -save_ascii_pssm options.'), _Option(['-in_msa', 'in_msa'], 'File name of multiple sequence alignment to restart PSI-BLAST.\n\nIncompatible with: in_pssm, query', filename=True, equate=False), _Option(['-msa_master_idx', 'msa_master_idx'], 'Index of sequence to use as master in MSA.\n\nIndex (1-based) of sequence to use as the master in the multiple sequence alignment. If not specified, the first sequence is used.', equate=False), _Option(['-in_pssm', 'in_pssm'], 'PSI-BLAST checkpoint file.\n\nIncompatible with: in_msa, query, phi_pattern', filename=True, equate=False), _Option(['-pseudocount', 'pseudocount'], 'Pseudo-count value used when constructing PSSM.\n\nInteger. Default is zero.', equate=False), _Option(['-inclusion_ethresh', 'inclusion_ethresh'], 'E-value inclusion threshold for pairwise alignments (float, default 0.002).', equate=False), _Switch(['-ignore_msa_master', 'ignore_msa_master'], 'Ignore the master sequence when creating PSSM.\n\nRequires: in_msa\nIncompatible with: msa_master_idx, in_pssm, query, query_loc, phi_pattern'), _Option(['-phi_pattern', 'phi_pattern'], 'File name containing pattern to search.\n\nIncompatible with: in_pssm', filename=True, equate=False)]
        _Ncbiblast2SeqCommandline.__init__(self, cmd, **kwargs)

    def _validate(self):
        incompatibles = {'num_iterations': ['remote'], 'in_msa': ['in_pssm', 'query'], 'in_pssm': ['in_msa', 'query', 'phi_pattern'], 'ignore_msa_master': ['msa_master_idx', 'in_pssm', 'query', 'query_loc', 'phi_pattern']}
        self._validate_incompatibilities(incompatibles)
        _Ncbiblast2SeqCommandline._validate(self)