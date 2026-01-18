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
def visit_footnote(self, node):
    self.footnote_level += 1
    self.save_footnote_current = self.current_element
    el1 = Element('text:note-body')
    self.current_element = el1
    self.footnote_list.append((node, el1))
    if isinstance(node, docutils.nodes.citation):
        self.paragraph_style_stack.append(self.rststyle('citation'))
    else:
        self.paragraph_style_stack.append(self.rststyle('footnote'))