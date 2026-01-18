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
def update_toc_page_numbers(self, el):
    collection = []
    self.update_toc_collect(el, 0, collection)
    self.update_toc_add_numbers(collection)