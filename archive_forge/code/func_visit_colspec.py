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
def visit_colspec(self, node):
    self.column_count += 1
    colspec_name = self.rststyle('%s%%d.%%s' % TABLESTYLEPREFIX, (self.table_count, chr(self.column_count)))
    colwidth = node['colwidth'] / 12.0
    el1 = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': colspec_name, 'style:family': 'table-column'}, nsdict=SNSD)
    SubElement(el1, 'style:table-column-properties', attrib={'style:column-width': '%.4fin' % colwidth}, nsdict=SNSD)
    self.append_child('table:table-column', attrib={'table:style-name': colspec_name})
    self.table_width += colwidth