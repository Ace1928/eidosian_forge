from kivy.event import EventDispatcher
import string
def _move_cursor_word_right(self, index=None):
    pos = index or self.cursor_index()
    col, row = self.get_cursor_from_index(pos)
    lines = self._lines
    mrow = len(lines) - 1
    if row == mrow and col == len(lines[row]):
        return (col, row)
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
    elif c in lcase:
        mode = 'camel'
    while True:
        if mode in ('normal', 'camel', 'punct') and c in ws:
            mode = 'ws'
        elif mode in ('normal', 'camel') and c == '_':
            mode = 'us'
        elif mode == 'normal' and c not in ucase:
            mode = 'camel'
        if mode == 'us':
            if c in ws:
                mode = 'ws'
            elif c != '_':
                break
        if mode == 'ws' and c not in ws:
            break
        if mode == 'camel' and c in ucase:
            break
        if mode == 'punct' and (c == '_' or c not in punct):
            break
        if mode != 'punct' and c != '_' and (c in punct):
            break
        col += 1
        if col > len(rline):
            if row == mrow:
                return (len(rline), mrow)
            row += 1
            rline = lines[row]
            col = 0
        c = rline[col] if len(rline) > col else '\n'
        if c == '\n':
            break
    return (col, row)