import Bio.GenBank
def _journal_line(self):
    """Output for JOURNAL information (PRIVATE)."""
    output = ''
    if self.journal:
        output += Record.INTERNAL_FORMAT % 'JOURNAL'
        output += _wrapped_genbank(self.journal, Record.GB_BASE_INDENT)
    return output