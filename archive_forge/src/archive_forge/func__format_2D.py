import os
import string
import numpy as np
from Bio.File import as_handle
def _format_2D(self, fmt):
    alphabet = self.alphabet
    n = len(alphabet)
    words = [[None] * n for _ in range(n)]
    lines = []
    try:
        header = self.header
    except AttributeError:
        pass
    else:
        for line in header:
            line = '#  %s\n' % line
            lines.append(line)
    keywidth = max((len(c) for c in alphabet))
    keyfmt = '%' + str(keywidth) + 's'
    line = ' ' * keywidth
    for j, c2 in enumerate(alphabet):
        maxwidth = 0
        for i, c1 in enumerate(alphabet):
            key = (c1, c2)
            value = self[key]
            word = fmt % value
            width = len(word)
            if width > maxwidth:
                maxwidth = width
            words[i][j] = word
        fmt2 = ' %' + str(maxwidth) + 's'
        word = fmt2 % c2
        line += word
        for i, c1 in enumerate(alphabet):
            word = words[i][j]
            words[i][j] = fmt2 % word
    line = line.rstrip() + '\n'
    lines.append(line)
    for letter, row in zip(alphabet, words):
        key = keyfmt % letter
        line = key + ''.join(row) + '\n'
        lines.append(line)
    text = ''.join(lines)
    return text