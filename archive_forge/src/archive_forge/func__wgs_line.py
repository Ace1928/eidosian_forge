import Bio.GenBank
def _wgs_line(self):
    output = ''
    if self.wgs:
        output += Record.BASE_FORMAT % 'WGS'
        output += self.wgs
    return output