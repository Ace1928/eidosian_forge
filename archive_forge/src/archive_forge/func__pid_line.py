import Bio.GenBank
def _pid_line(self):
    """Output for PID line. Presumedly, PID usage is also obsolete (PRIVATE)."""
    if self.pid:
        output = Record.BASE_FORMAT % 'PID'
        output += f'{self.pid}\n'
    else:
        output = ''
    return output