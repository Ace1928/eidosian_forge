import re
from mako import exceptions
def _is_unindentor(self, line):
    """return true if the given line is an 'unindentor',
        relative to the last 'indent' event received.

        """
    if len(self.indent_detail) == 0:
        return False
    indentor = self.indent_detail[-1]
    if indentor is None:
        return False
    match = self._re_unindentor.match(line)
    return bool(match)