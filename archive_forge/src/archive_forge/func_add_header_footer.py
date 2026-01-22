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
def add_header_footer(self, root_el):
    automatic_styles = root_el.find('{%s}automatic-styles' % SNSD['office'])
    path = '{%s}master-styles' % (NAME_SPACE_1,)
    master_el = root_el.find(path)
    if master_el is None:
        return
    path = '{%s}master-page' % (SNSD['style'],)
    master_el_container = master_el.findall(path)
    master_el = None
    target_attrib = '{%s}name' % (SNSD['style'],)
    target_name = self.rststyle('pagedefault')
    for el in master_el_container:
        if el.get(target_attrib) == target_name:
            master_el = el
            break
    if master_el is None:
        return
    el1 = master_el
    if self.header_content or self.settings.custom_header:
        if WhichElementTree == 'lxml':
            el2 = SubElement(el1, 'style:header', nsdict=SNSD)
        else:
            el2 = SubElement(el1, 'style:header', attrib=STYLES_NAMESPACE_ATTRIB, nsdict=STYLES_NAMESPACE_DICT)
        for el in self.header_content:
            attrkey = add_ns('text:style-name', nsdict=SNSD)
            el.attrib[attrkey] = self.rststyle('header')
            el2.append(el)
        if self.settings.custom_header:
            self.create_custom_headfoot(el2, self.settings.custom_header, 'header', automatic_styles)
    if self.footer_content or self.settings.custom_footer:
        if WhichElementTree == 'lxml':
            el2 = SubElement(el1, 'style:footer', nsdict=SNSD)
        else:
            el2 = SubElement(el1, 'style:footer', attrib=STYLES_NAMESPACE_ATTRIB, nsdict=STYLES_NAMESPACE_DICT)
        for el in self.footer_content:
            attrkey = add_ns('text:style-name', nsdict=SNSD)
            el.attrib[attrkey] = self.rststyle('footer')
            el2.append(el)
        if self.settings.custom_footer:
            self.create_custom_headfoot(el2, self.settings.custom_footer, 'footer', automatic_styles)