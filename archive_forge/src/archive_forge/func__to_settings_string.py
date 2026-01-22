import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def _to_settings_string(node, l, **kwargs):
    tag = node.tag
    text = node.text
    if len(node) == 0:
        if text:
            l.append(f'\\{tag} {text}\n')
        else:
            l.append('\\%s\n' % tag)
    else:
        if text:
            l.append(f'\\+{tag} {text}\n')
        else:
            l.append('\\+%s\n' % tag)
        for n in node:
            _to_settings_string(n, l, **kwargs)
        l.append('\\-%s\n' % tag)
    return