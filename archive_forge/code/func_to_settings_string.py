import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def to_settings_string(tree, encoding=None, errors='strict', unicode_fields=None):
    l = list()
    _to_settings_string(tree.getroot(), l, encoding=encoding, errors=errors, unicode_fields=unicode_fields)
    return ''.join(l)