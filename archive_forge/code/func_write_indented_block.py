import re
from mako import exceptions
def write_indented_block(self, block, starting_lineno=None):
    """print a line or lines of python which already contain indentation.

        The indentation of the total block of lines will be adjusted to that of
        the current indent level."""
    self.in_indent_lines = False
    for i, l in enumerate(re.split('\\r?\\n', block)):
        self.line_buffer.append(l)
        if starting_lineno is not None:
            self.start_source(starting_lineno + i)
        self._update_lineno(1)