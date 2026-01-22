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
class _ElementInterfaceWrapper(_ElementInterface):

    def __init__(self, tag, attrib=None):
        _ElementInterface.__init__(self, tag, attrib)
        _parents[self] = None

    def setparent(self, parent):
        _parents[self] = parent

    def getparent(self):
        return _parents[self]