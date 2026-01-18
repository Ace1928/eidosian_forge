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
def visit_superscript(self, node):
    el = self.append_child('text:span', attrib={'text:style-name': 'rststyle-superscript'})
    self.set_current_element(el)