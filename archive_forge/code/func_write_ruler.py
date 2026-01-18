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
def write_ruler(outfile):
    outfile.write('+')
    outfile.write('-' * 7)
    outfile.write('+')
    outfile.write('-' * 68)
    outfile.write('+')
    outfile.write('\n')