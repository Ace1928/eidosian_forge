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
def visit_line(self, node):
    style = 'lineblock%d' % self.line_indent_level
    el1 = SubElement(self.current_element, 'text:p', attrib={'text:style-name': self.rststyle(style)})
    self.current_element = el1