import sys, string, re
import getopt
from distutils.errors import *
def wrap_text(text, width):
    """wrap_text(text : string, width : int) -> [string]

    Split 'text' into multiple lines of no more than 'width' characters
    each, and return the list of strings that results.
    """
    if text is None:
        return []
    if len(text) <= width:
        return [text]
    text = text.expandtabs()
    text = text.translate(WS_TRANS)
    chunks = re.split('( +|-+)', text)
    chunks = [ch for ch in chunks if ch]
    lines = []
    while chunks:
        cur_line = []
        cur_len = 0
        while chunks:
            l = len(chunks[0])
            if cur_len + l <= width:
                cur_line.append(chunks[0])
                del chunks[0]
                cur_len = cur_len + l
            else:
                if cur_line and cur_line[-1][0] == ' ':
                    del cur_line[-1]
                break
        if chunks:
            if cur_len == 0:
                cur_line.append(chunks[0][0:width])
                chunks[0] = chunks[0][width:]
            if chunks[0][0] == ' ':
                del chunks[0]
        lines.append(''.join(cur_line))
    return lines