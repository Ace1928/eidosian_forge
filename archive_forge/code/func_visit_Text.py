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
def visit_Text(self, node):
    if isinstance(node.parent, docutils.nodes.literal_block):
        return
    text = node.astext()
    if len(self.current_element.getchildren()) > 0:
        if self.current_element.getchildren()[-1].tail:
            self.current_element.getchildren()[-1].tail += text
        else:
            self.current_element.getchildren()[-1].tail = text
    elif self.current_element.text:
        self.current_element.text += text
    else:
        self.current_element.text = text