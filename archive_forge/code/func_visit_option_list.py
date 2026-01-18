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
def visit_option_list(self, node):
    table_name = 'tableoption'
    if not self.optiontablestyles_generated:
        self.optiontablestyles_generated = True
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle(table_name), 'style:family': 'table'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-properties', attrib={'style:width': '17.59cm', 'table:align': 'left', 'style:shadow': 'none'}, nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c' % table_name, ('A',)), 'style:family': 'table-column'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-column-properties', attrib={'style:column-width': '4.999cm'}, nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c' % table_name, ('B',)), 'style:family': 'table-column'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-column-properties', attrib={'style:column-width': '12.587cm'}, nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c%%d' % table_name, ('A', 1)), 'style:family': 'table-cell'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-cell-properties', attrib={'fo:background-color': 'transparent', 'fo:padding': '0.097cm', 'fo:border-left': '0.035cm solid #000000', 'fo:border-right': 'none', 'fo:border-top': '0.035cm solid #000000', 'fo:border-bottom': '0.035cm solid #000000'}, nsdict=SNSD)
        el2 = SubElement(el1, 'style:background-image', nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c%%d' % table_name, ('B', 1)), 'style:family': 'table-cell'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-cell-properties', attrib={'fo:padding': '0.097cm', 'fo:border': '0.035cm solid #000000'}, nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c%%d' % table_name, ('A', 2)), 'style:family': 'table-cell'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-cell-properties', attrib={'fo:padding': '0.097cm', 'fo:border-left': '0.035cm solid #000000', 'fo:border-right': 'none', 'fo:border-top': 'none', 'fo:border-bottom': '0.035cm solid #000000'}, nsdict=SNSD)
        el = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': self.rststyle('%s.%%c%%d' % table_name, ('B', 2)), 'style:family': 'table-cell'}, nsdict=SNSD)
        el1 = SubElement(el, 'style:table-cell-properties', attrib={'fo:padding': '0.097cm', 'fo:border-left': '0.035cm solid #000000', 'fo:border-right': '0.035cm solid #000000', 'fo:border-top': 'none', 'fo:border-bottom': '0.035cm solid #000000'}, nsdict=SNSD)
    el = self.append_child('table:table', attrib={'table:name': self.rststyle(table_name), 'table:style-name': self.rststyle(table_name)})
    el1 = SubElement(el, 'table:table-column', attrib={'table:style-name': self.rststyle('%s.%%c' % table_name, ('A',))})
    el1 = SubElement(el, 'table:table-column', attrib={'table:style-name': self.rststyle('%s.%%c' % table_name, ('B',))})
    el1 = SubElement(el, 'table:table-header-rows')
    el2 = SubElement(el1, 'table:table-row')
    el3 = SubElement(el2, 'table:table-cell', attrib={'table:style-name': self.rststyle('%s.%%c%%d' % table_name, ('A', 1)), 'office:value-type': 'string'})
    el4 = SubElement(el3, 'text:p', attrib={'text:style-name': 'Table_20_Heading'})
    el4.text = 'Option'
    el3 = SubElement(el2, 'table:table-cell', attrib={'table:style-name': self.rststyle('%s.%%c%%d' % table_name, ('B', 1)), 'office:value-type': 'string'})
    el4 = SubElement(el3, 'text:p', attrib={'text:style-name': 'Table_20_Heading'})
    el4.text = 'Description'
    self.set_current_element(el)