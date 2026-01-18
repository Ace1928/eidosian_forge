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
def visit_option_group(self, node):
    el = self.append_child('table:table-cell', attrib={'table:style-name': 'Table%d.A2' % self.table_count, 'office:value-type': 'string'})
    self.set_current_element(el)