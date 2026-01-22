from kivy.event import EventDispatcher
import string
def _move_cursor_word_left(self, index=None):
    pos = index or self.cursor_index()
    pos -= 1
    if pos == 0:
        return (0, 0)
    col, row = self.get_cursor_from_index(pos)
    lines = self._lines
    ucase = string.ascii_uppercase
    lcase = string.ascii_lowercase
    ws = string.whitespace
    punct = string.punctuation
    mode = 'normal'
    rline = lines[row]
    c = rline[col] if len(rline) > col else '\n'
    if c in ws:
        mode = 'ws'
    elif c == '_':
        mode = 'us'
    elif c in punct:
        mode = 'punct'
    elif c not in ucase:
        mode = 'camel'
    while True:
        if col == -1:
            if row == 0:
                return (0, 0)
            row -= 1
            rline = lines[row]
            col = len(rline)
        lc = c
        c = rline[col] if len(rline) > col else '\n'
        if c == '\n':
            if lc not in ws:
                col += 1
            break
        if mode in ('normal', 'camel') and c in ws:
            col += 1
            break
        if mode in ('normal', 'camel') and c in punct:
            col += 1
            break
        if mode == 'camel' and c in ucase:
            break
        if mode == 'punct' and (c == '_' or c not in punct):
            col += 1
            break
        if mode == 'us' and c != '_' and (c in punct or c in ws):
            col += 1
            break
        if mode == 'us' and c != '_':
            mode = 'normal' if c in ucase else 'ws' if c in ws else 'camel'
        elif mode == 'ws' and c not in ws:
            mode = 'normal' if c in ucase else 'us' if c == '_' else 'punct' if c in punct else 'camel'
        col -= 1
    if col > len(rline):
        if row == len(lines) - 1:
            return (row, len(lines[row]))
        row += 1
        col = 0
    return (col, row)