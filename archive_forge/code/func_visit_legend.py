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
def visit_legend(self, node):
    if isinstance(node.parent, docutils.nodes.figure):
        el1 = self.current_element[-1]
        el1 = el1[0][0]
        self.current_element = el1
        self.paragraph_style_stack.append(self.rststyle('legend'))