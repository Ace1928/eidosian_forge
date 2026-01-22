import Bio.GenBank
def _wgs_scafld_line(self):
    output = ''
    if self.wgs_scafld:
        output += Record.BASE_FORMAT % 'WGS_SCAFLD'
        output += self.wgs_scafld
    return output