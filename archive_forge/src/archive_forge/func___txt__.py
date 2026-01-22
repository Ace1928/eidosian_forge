from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
def __txt__(self, startindex):
    columns = self.header
    if not columns:
        columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
    if not self.columns_len or len(self.columns_len) != len(columns):
        self.columns_len = [len(c) for c in columns]
    chapters_txt = {}
    offsets = defaultdict(int)
    for name, chapter in self.chapters.items():
        chapters_txt[name] = chapter.__txt__(startindex)
        if startindex == 0:
            offsets[name] = len(chapters_txt[name]) - len(self)
    str_matrix = []
    for i, line in enumerate(self[startindex:]):
        str_line = []
        for j, name in enumerate(columns):
            if name in chapters_txt:
                column = chapters_txt[name][i + offsets[name]]
            else:
                value = line.get(name, '')
                string = '{0:n}' if isinstance(value, float) else '{0}'
                column = string.format(value)
            self.columns_len[j] = max(self.columns_len[j], len(column))
            str_line.append(column)
        str_matrix.append(str_line)
    if startindex == 0 and self.log_header:
        header = []
        nlines = 1
        if len(self.chapters) > 0:
            nlines += max(map(len, chapters_txt.values())) - len(self) + 1
        header = [[] for i in range(nlines)]
        for j, name in enumerate(columns):
            if name in chapters_txt:
                length = max((len(line.expandtabs()) for line in chapters_txt[name]))
                blanks = nlines - 2 - offsets[name]
                for i in range(blanks):
                    header[i].append(' ' * length)
                header[blanks].append(name.center(length))
                header[blanks + 1].append('-' * length)
                for i in range(offsets[name]):
                    header[blanks + 2 + i].append(chapters_txt[name][i])
            else:
                length = max((len(line[j].expandtabs()) for line in str_matrix))
                for line in header[:-1]:
                    line.append(' ' * length)
                header[-1].append(name)
        str_matrix = chain(header, str_matrix)
    template = '\t'.join(('{%i:<%i}' % (i, l) for i, l in enumerate(self.columns_len)))
    text = [template.format(*line) for line in str_matrix]
    return text