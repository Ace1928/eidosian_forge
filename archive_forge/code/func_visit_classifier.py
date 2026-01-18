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
def visit_classifier(self, node):
    els = self.current_element.getchildren()
    if len(els) > 0:
        el = els[-1]
        el1 = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle('emphasis')})
        el1.text = ' (%s)' % (node.astext(),)