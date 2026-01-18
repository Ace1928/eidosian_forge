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
def visit_math_block(self, node):
    self.document.reporter.warning('"math" directive not supported', base_node=node)
    self.visit_literal_block(node)