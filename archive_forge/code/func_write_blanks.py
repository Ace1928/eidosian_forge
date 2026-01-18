import re
from mako import exceptions
def write_blanks(self, num):
    self.stream.write('\n' * num)
    self._update_lineno(num)