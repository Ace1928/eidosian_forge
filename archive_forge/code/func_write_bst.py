from __future__ import print_function, unicode_literals, with_statement
import re
import sys
from os import path
from shutil import rmtree
from subprocess import PIPE, Popen
from tempfile import mkdtemp
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.output import bibtex
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
def write_bst(filename, style):
    with open(filename, 'w') as bst_file:
        bst_file.write(style)
        bst_file.write('\n')