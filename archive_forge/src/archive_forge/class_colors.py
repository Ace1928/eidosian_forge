from __future__ import division, print_function, unicode_literals
import re
import timeit
import codecs
import argparse
import sys
from builtins import str
from commonmark.render.html import HtmlRenderer
from commonmark.main import Parser, dumpAST
class colors(object):
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'