import re
import docutils
from docutils import nodes, writers, languages
class enum_char(object):
    enum_style = {'bullet': '\\(bu', 'emdash': '\\(em'}

    def __init__(self, style):
        self._style = style
        if 'start' in node:
            self._cnt = node['start'] - 1
        else:
            self._cnt = 0
        self._indent = 2
        if style == 'arabic':
            self._indent = len(str(len(node.children)))
            self._indent += len(str(self._cnt)) + 1
        elif style == 'loweralpha':
            self._cnt += ord('a') - 1
            self._indent = 3
        elif style == 'upperalpha':
            self._cnt += ord('A') - 1
            self._indent = 3
        elif style.endswith('roman'):
            self._indent = 5

    def __next__(self):
        if self._style == 'bullet':
            return self.enum_style[self._style]
        elif self._style == 'emdash':
            return self.enum_style[self._style]
        self._cnt += 1
        if self._style == 'arabic':
            return '%d.' % self._cnt
        elif self._style in ('loweralpha', 'upperalpha'):
            return '%c.' % self._cnt
        elif self._style.endswith('roman'):
            res = roman.toRoman(self._cnt) + '.'
            if self._style.startswith('upper'):
                return res.upper()
            return res.lower()
        else:
            return '%d.' % self._cnt

    def get_width(self):
        return self._indent

    def __repr__(self):
        return 'enum_style-%s' % list(self._style)