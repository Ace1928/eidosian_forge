import Bio.GenBank
def _pubmed_line(self):
    """Output for PUBMED information (PRIVATE)."""
    output = ''
    if self.pubmed_id:
        output += Record.OTHER_INTERNAL_FORMAT % 'PUBMED'
        output += self.pubmed_id + '\n'
    return output