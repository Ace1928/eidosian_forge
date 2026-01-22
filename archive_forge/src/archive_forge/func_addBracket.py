from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def addBracket(self, node, index, image):
    if self.brackets is not None:
        self.brackets['bracketAfter'] = True
    self.brackets = {'node': node, 'previous': self.brackets, 'previousDelimiter': self.delimiters, 'index': index, 'image': image, 'active': True}