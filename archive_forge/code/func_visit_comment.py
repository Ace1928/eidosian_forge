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
def visit_comment(self, node):
    el = self.append_p('textbody')
    el1 = SubElement(el, 'office:annotation', attrib={})
    el2 = SubElement(el1, 'dc:creator', attrib={})
    s1 = os.environ.get('USER', '')
    el2.text = s1
    el2 = SubElement(el1, 'text:p', attrib={})
    el2.text = node.astext()