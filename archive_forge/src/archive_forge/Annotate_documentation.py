from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils

        lines : original cython source code split by lines
        generated_code : generated c code keyed by line number in original file
        target filename : name of the file in which to store the generated html
        c_file : filename in which the c_code has been written
        