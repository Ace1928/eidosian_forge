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
def visit_label(self, node):
    if isinstance(node.parent, docutils.nodes.footnote):
        raise nodes.SkipChildren()
    elif self.citation_id is not None:
        el = self.append_p('textbody')
        self.set_current_element(el)
        if self.settings.create_links:
            el0 = SubElement(el, 'text:span')
            el0.text = '['
            self.append_child('text:reference-mark-start', attrib={'text:name': '%s' % (self.citation_id,)})
        else:
            el.text = '['