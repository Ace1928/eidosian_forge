from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import textwrap
import cmakelang
from cmakelang.lint import lintdb
from tangent.tooling.gendoc import format_directive
def write_cell(outfile, idstr, msgfmt):
    lines = textwrap.wrap(msgfmt, width=66)
    outfile.write('|`{:5s}`| {:66s} |\n'.format(idstr, lines.pop(0)))
    for line in lines:
        outfile.write('| {:5s} | {:66s} |\n'.format('', line))