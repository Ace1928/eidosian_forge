from Bio.Application import _Option, AbstractCommandline, _Switch
class _NcbibaseblastCommandline(AbstractCommandline):
    """Base Commandline object for (new) NCBI BLAST+ wrappers (PRIVATE).

    This is provided for subclassing, it deals with shared options
    common to all the BLAST tools (blastn, rpsblast, rpsblast, etc
    AND blast_formatter).
    """

    def __init__(self, cmd=None, **kwargs):
        assert cmd is not None
        extra_parameters = [_Switch(['-h', 'h'], 'Print USAGE and DESCRIPTION;  ignore other arguments.'), _Switch(['-help', 'help'], 'Print USAGE, DESCRIPTION and ARGUMENTS description; ignore other arguments.'), _Switch(['-version', 'version'], 'Print version number;  ignore other arguments.'), _Option(['-out', 'out'], 'Output file for alignment.', filename=True, equate=False), _Option(['-outfmt', 'outfmt'], "Alignment view.  Typically an integer 0-14 but for some formats can be named columns like '6 qseqid sseqid'.  Use 5 for XML output (differs from classic BLAST which used 7 for XML).", filename=True, equate=False), _Switch(['-show_gis', 'show_gis'], 'Show NCBI GIs in deflines?'), _Option(['-num_descriptions', 'num_descriptions'], 'Number of database sequences to show one-line descriptions for.\n\nInteger argument (at least zero). Default is 500. See also num_alignments.', equate=False), _Option(['-num_alignments', 'num_alignments'], 'Number of database sequences to show num_alignments for.\n\nInteger argument (at least zero). Default is 200. See also num_alignments.', equate=False), _Option(['-line_length', 'line_length'], 'Line length for formatting alignments (integer, at least 1, default 60).\n\nNot applicable for outfmt > 4. Added in BLAST+ 2.2.30.', equate=False), _Switch(['-html', 'html'], 'Produce HTML output? See also the outfmt option.'), _Switch(['-parse_deflines', 'parse_deflines'], 'Should the query and subject defline(s) be parsed?')]
        try:
            self.parameters = extra_parameters + self.parameters
        except AttributeError:
            self.parameters = extra_parameters
        AbstractCommandline.__init__(self, cmd, **kwargs)

    def _validate_incompatibilities(self, incompatibles):
        """Validate parameters for incompatibilities (PRIVATE).

        Used by the _validate method.
        """
        for a in incompatibles:
            if self._get_parameter(a):
                for b in incompatibles[a]:
                    if self._get_parameter(b):
                        raise ValueError(f'Options {a} and {b} are incompatible.')