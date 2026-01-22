import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _wiki_table_sub(self, match):
    ttext = match.group(0).strip()
    rows = []
    for line in ttext.splitlines(0):
        line = line.strip()[2:-2].strip()
        row = [c.strip() for c in re.split('(?<!\\\\)\\|\\|', line)]
        rows.append(row)
    hlines = []

    def add_hline(line, indents=0):
        hlines.append(self.tab * indents + line)

    def format_cell(text):
        return self._run_span_gamut(re.sub('^\\s*~', '', cell).strip(' '))
    add_hline('<table%s>' % self._html_class_str_from_tag('table'))
    if rows and rows[0] and re.match('^\\s*~', rows[0][0]):
        add_hline('<thead%s>' % self._html_class_str_from_tag('thead'), 1)
        add_hline('<tr>', 2)
        for cell in rows[0]:
            add_hline('<th>{}</th>'.format(format_cell(cell)), 3)
        add_hline('</tr>', 2)
        add_hline('</thead>', 1)
        rows = rows[1:]
    if rows:
        add_hline('<tbody>', 1)
        for row in rows:
            add_hline('<tr>', 2)
            for cell in row:
                add_hline('<td>{}</td>'.format(format_cell(cell)), 3)
            add_hline('</tr>', 2)
        add_hline('</tbody>', 1)
    add_hline('</table>')
    return '\n'.join(hlines) + '\n'