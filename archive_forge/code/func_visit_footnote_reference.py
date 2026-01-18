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
def visit_footnote_reference(self, node):
    if self.footnote_level <= 0:
        id = node.attributes['ids'][0]
        refid = node.attributes.get('refid')
        if refid is None:
            refid = ''
        if self.settings.endnotes_end_doc:
            note_class = 'endnote'
        else:
            note_class = 'footnote'
        el1 = self.append_child('text:note', attrib={'text:id': '%s' % (refid,), 'text:note-class': note_class})
        note_auto = str(node.attributes.get('auto', 1))
        if isinstance(node, docutils.nodes.citation_reference):
            citation = '[%s]' % node.astext()
            el2 = SubElement(el1, 'text:note-citation', attrib={'text:label': citation})
            el2.text = citation
        elif note_auto == '1':
            el2 = SubElement(el1, 'text:note-citation', attrib={'text:label': node.astext()})
            el2.text = node.astext()
        elif note_auto == '*':
            if self.footnote_chars_idx >= len(ODFTranslator.footnote_chars):
                self.footnote_chars_idx = 0
            footnote_char = ODFTranslator.footnote_chars[self.footnote_chars_idx]
            self.footnote_chars_idx += 1
            el2 = SubElement(el1, 'text:note-citation', attrib={'text:label': footnote_char})
            el2.text = footnote_char
        self.footnote_ref_dict[id] = el1
    raise nodes.SkipChildren()