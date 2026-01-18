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
def visit_option(self, node):
    el = self.append_child('text:p', attrib={'text:style-name': 'Table_20_Contents'})
    el.text = node.astext()