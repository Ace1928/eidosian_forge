import Bio.GenBank
def _organism_line(self):
    """Output for ORGANISM line with taxonomy info (PRIVATE)."""
    output = Record.INTERNAL_FORMAT % 'ORGANISM'
    output += _wrapped_genbank(self.organism, Record.GB_BASE_INDENT)
    output += ' ' * Record.GB_BASE_INDENT
    taxonomy_info = ''
    for tax in self.taxonomy:
        taxonomy_info += f'{tax}; '
    taxonomy_info = taxonomy_info[:-2]
    taxonomy_info += '.'
    output += _wrapped_genbank(taxonomy_info, Record.GB_BASE_INDENT)
    return output