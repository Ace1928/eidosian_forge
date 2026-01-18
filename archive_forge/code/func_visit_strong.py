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
def visit_strong(self, node):
    el = SubElement(self.current_element, 'text:span', attrib={'text:style-name': self.rststyle('strong')})
    self.set_current_element(el)