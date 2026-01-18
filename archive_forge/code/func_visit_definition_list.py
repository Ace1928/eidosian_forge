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
def visit_definition_list(self, node):
    self.def_list_level += 1
    if self.list_level > 5:
        raise RuntimeError('max definition list nesting level exceeded')