import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
class SimpleTableParser(TableParser):
    """
    Parse a simple table using `parse()`.

    Here's an example of a simple table::

        =====  =====
        col 1  col 2
        =====  =====
        1      Second column of row 1.
        2      Second column of row 2.
               Second line of paragraph.
        3      - Second column of row 3.

               - Second item in bullet
                 list (row 3, column 2).
        4 is a span
        ------------
        5
        =====  =====

    Top and bottom borders use '=', column span underlines use '-', column
    separation is indicated with spaces.

    Passing the above table to the `parse()` method will result in the
    following data structure, whose interpretation is the same as for
    `GridTableParser`::

        ([5, 25],
         [[(0, 0, 1, ['col 1']),
           (0, 0, 1, ['col 2'])]],
         [[(0, 0, 3, ['1']),
           (0, 0, 3, ['Second column of row 1.'])],
          [(0, 0, 4, ['2']),
           (0, 0, 4, ['Second column of row 2.',
                      'Second line of paragraph.'])],
          [(0, 0, 6, ['3']),
           (0, 0, 6, ['- Second column of row 3.',
                      '',
                      '- Second item in bullet',
                      '  list (row 3, column 2).'])],
          [(0, 1, 10, ['4 is a span'])],
          [(0, 0, 12, ['5']),
           (0, 0, 12, [''])]])
    """
    head_body_separator_pat = re.compile('=[ =]*$')
    span_pat = re.compile('-[ -]*$')

    def setup(self, block):
        self.block = block[:]
        self.block.disconnect()
        self.block[0] = self.block[0].replace('=', '-')
        self.block[-1] = self.block[-1].replace('=', '-')
        self.head_body_sep = None
        self.columns = []
        self.border_end = None
        self.table = []
        self.done = [-1] * len(block[0])
        self.rowseps = {0: [0]}
        self.colseps = {0: [0]}

    def parse_table(self):
        """
        First determine the column boundaries from the top border, then
        process rows.  Each row may consist of multiple lines; accumulate
        lines until a row is complete.  Call `self.parse_row` to finish the
        job.
        """
        self.columns = self.parse_columns(self.block[0], 0)
        self.border_end = self.columns[-1][1]
        firststart, firstend = self.columns[0]
        offset = 1
        start = 1
        text_found = None
        while offset < len(self.block):
            line = self.block[offset]
            if self.span_pat.match(line):
                self.parse_row(self.block[start:offset], start, (line.rstrip(), offset))
                start = offset + 1
                text_found = None
            elif line[firststart:firstend].strip():
                if text_found and offset != start:
                    self.parse_row(self.block[start:offset], start)
                start = offset
                text_found = 1
            elif not text_found:
                start = offset + 1
            offset += 1

    def parse_columns(self, line, offset):
        """
        Given a column span underline, return a list of (begin, end) pairs.
        """
        cols = []
        end = 0
        while True:
            begin = line.find('-', end)
            end = line.find(' ', begin)
            if begin < 0:
                break
            if end < 0:
                end = len(line)
            cols.append((begin, end))
        if self.columns:
            if cols[-1][1] != self.border_end:
                raise TableMarkupError('Column span incomplete in table line %s.' % (offset + 1), offset=offset)
            cols[-1] = (cols[-1][0], self.columns[-1][1])
        return cols

    def init_row(self, colspec, offset):
        i = 0
        cells = []
        for start, end in colspec:
            morecols = 0
            try:
                assert start == self.columns[i][0]
                while end != self.columns[i][1]:
                    i += 1
                    morecols += 1
            except (AssertionError, IndexError):
                raise TableMarkupError('Column span alignment problem in table line %s.' % (offset + 2), offset=offset + 1)
            cells.append([0, morecols, offset, []])
            i += 1
        return cells

    def parse_row(self, lines, start, spanline=None):
        """
        Given the text `lines` of a row, parse it and append to `self.table`.

        The row is parsed according to the current column spec (either
        `spanline` if provided or `self.columns`).  For each column, extract
        text from each line, and check for text in column margins.  Finally,
        adjust for insignificant whitespace.
        """
        if not (lines or spanline):
            return
        if spanline:
            columns = self.parse_columns(*spanline)
            span_offset = spanline[1]
        else:
            columns = self.columns[:]
            span_offset = start
        self.check_columns(lines, start, columns)
        row = self.init_row(columns, start)
        for i in range(len(columns)):
            start, end = columns[i]
            cellblock = lines.get_2D_block(0, start, len(lines), end)
            cellblock.disconnect()
            cellblock.replace(self.double_width_pad_char, '')
            row[i][3] = cellblock
        self.table.append(row)

    def check_columns(self, lines, first_line, columns):
        """
        Check for text in column margins and text overflow in the last column.
        Raise TableMarkupError if anything but whitespace is in column margins.
        Adjust the end value for the last column if there is text overflow.
        """
        columns.append((sys.maxsize, None))
        lastcol = len(columns) - 2
        lines = [strip_combining_chars(line) for line in lines]
        for i in range(len(columns) - 1):
            start, end = columns[i]
            nextstart = columns[i + 1][0]
            offset = 0
            for line in lines:
                if i == lastcol and line[end:].strip():
                    text = line[start:].rstrip()
                    new_end = start + len(text)
                    main_start, main_end = self.columns[-1]
                    columns[i] = (start, max(main_end, new_end))
                    if new_end > main_end:
                        self.columns[-1] = (main_start, new_end)
                elif line[end:nextstart].strip():
                    raise TableMarkupError('Text in column margin in table line %s.' % (first_line + offset + 1), offset=first_line + offset)
                offset += 1
        columns.pop()

    def structure_from_cells(self):
        colspecs = [end - start for start, end in self.columns]
        first_body_row = 0
        if self.head_body_sep:
            for i in range(len(self.table)):
                if self.table[i][0][2] > self.head_body_sep:
                    first_body_row = i
                    break
        return (colspecs, self.table[:first_body_row], self.table[first_body_row:])