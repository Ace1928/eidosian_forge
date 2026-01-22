import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
class TableParser:
    """
    Abstract superclass for the common parts of the syntax-specific parsers.
    """
    head_body_separator_pat = None
    'Matches the row separator between head rows and body rows.'
    double_width_pad_char = '\x00'
    'Padding character for East Asian double-width text.'

    def parse(self, block):
        """
        Analyze the text `block` and return a table data structure.

        Given a plaintext-graphic table in `block` (list of lines of text; no
        whitespace padding), parse the table, construct and return the data
        necessary to construct a CALS table or equivalent.

        Raise `TableMarkupError` if there is any problem with the markup.
        """
        self.setup(block)
        self.find_head_body_sep()
        self.parse_table()
        structure = self.structure_from_cells()
        return structure

    def find_head_body_sep(self):
        """Look for a head/body row separator line; store the line index."""
        for i in range(len(self.block)):
            line = self.block[i]
            if self.head_body_separator_pat.match(line):
                if self.head_body_sep:
                    raise TableMarkupError('Multiple head/body row separators (table lines %s and %s); only one allowed.' % (self.head_body_sep + 1, i + 1), offset=i)
                else:
                    self.head_body_sep = i
                    self.block[i] = line.replace('=', '-')
        if self.head_body_sep == 0 or self.head_body_sep == len(self.block) - 1:
            raise TableMarkupError('The head/body row separator may not be the first or last line of the table.', offset=i)