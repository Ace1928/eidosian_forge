import Bio.GenBank
def _nid_line(self):
    """Output for the NID line. Use of NID is obsolete in GenBank files (PRIVATE)."""
    if self.nid:
        output = Record.BASE_FORMAT % 'NID'
        output += f'{self.nid}\n'
    else:
        output = ''
    return output