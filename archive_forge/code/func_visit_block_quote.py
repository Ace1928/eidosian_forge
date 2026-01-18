import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def visit_block_quote(self, node):
    if 'epigraph' in node.attributes['classes']:
        self.paragraph_style_stack.append(self.rststyle('epigraph'))
        self.blockstyle = self.rststyle('epigraph')
    elif 'highlights' in node.attributes['classes']:
        self.paragraph_style_stack.append(self.rststyle('highlights'))
        self.blockstyle = self.rststyle('highlights')
    else:
        self.paragraph_style_stack.append(self.rststyle('blockquote'))
        self.blockstyle = self.rststyle('blockquote')
    self.line_indent_level += 1