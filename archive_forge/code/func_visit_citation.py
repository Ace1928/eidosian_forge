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
def visit_citation(self, node):
    self.in_citation = True
    for id in node.attributes['ids']:
        self.citation_id = id
        break
    self.paragraph_style_stack.append(self.rststyle('blockindent'))
    self.bumped_list_level_stack.append(ListLevel(1))