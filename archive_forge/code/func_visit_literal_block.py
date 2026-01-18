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
def visit_literal_block(self, node):
    if len(self.paragraph_style_stack) > 1:
        wrapper1 = '<text:p text:style-name="%s">%%s</text:p>' % (self.rststyle('codeblock-indented'),)
    else:
        wrapper1 = '<text:p text:style-name="%s">%%s</text:p>' % (self.rststyle('codeblock'),)
    source = node.astext()
    if pygments and self.settings.add_syntax_highlighting:
        language = node.get('language', 'python')
        source = self._add_syntax_highlighting(source, language)
    else:
        source = escape_cdata(source)
    lines = source.split('\n')
    if lines[-1] == '':
        del lines[-1]
    lines1 = ['<wrappertag1 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">']
    my_lines = []
    for my_line in lines:
        my_line = self.fill_line(my_line)
        my_line = my_line.replace('&#10;', '\n')
        my_lines.append(my_line)
    my_lines_str = '<text:line-break/>'.join(my_lines)
    my_lines_str2 = wrapper1 % (my_lines_str,)
    lines1.append(my_lines_str2)
    lines1.append('</wrappertag1>')
    s1 = ''.join(lines1)
    if WhichElementTree != 'lxml':
        s1 = s1.encode('utf-8')
    el1 = etree.fromstring(s1)
    children = el1.getchildren()
    for child in children:
        self.current_element.append(child)