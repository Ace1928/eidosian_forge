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
def visit_table(self, node):
    self.table_count += 1
    table_style = self.get_table_style(node)
    table_name = '%s%%d' % TABLESTYLEPREFIX
    el1 = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s' % table_name, (self.table_count,)), 'style:family': 'table'}, nsdict=SNSD)
    if table_style.backgroundcolor is None:
        SubElement(el1, 'style:table-properties', attrib={'table:align': 'left', 'fo:margin-top': '0in', 'fo:margin-bottom': '0.10in'}, nsdict=SNSD)
    else:
        SubElement(el1, 'style:table-properties', attrib={'table:align': 'margins', 'fo:margin-top': '0in', 'fo:margin-bottom': '0.10in', 'fo:background-color': table_style.backgroundcolor}, nsdict=SNSD)
    el2 = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c%%d' % table_name, (self.table_count, 'A', 1)), 'style:family': 'table-cell'}, nsdict=SNSD)
    thickness = self.settings.table_border_thickness
    if thickness is None:
        line_style1 = table_style.border
    else:
        line_style1 = '0.%03dcm solid #000000' % (thickness,)
    SubElement(el2, 'style:table-cell-properties', attrib={'fo:padding': '0.049cm', 'fo:border-left': line_style1, 'fo:border-right': line_style1, 'fo:border-top': line_style1, 'fo:border-bottom': line_style1}, nsdict=SNSD)
    title = None
    for child in node.children:
        if child.tagname == 'title':
            title = child.astext()
            break
    if title is not None:
        self.append_p('table-title', title)
    else:
        pass
    el4 = SubElement(self.current_element, 'table:table', attrib={'table:name': self.rststyle('%s' % table_name, (self.table_count,)), 'table:style-name': self.rststyle('%s' % table_name, (self.table_count,))})
    self.set_current_element(el4)
    self.current_table_style = el1
    self.table_width = 0.0