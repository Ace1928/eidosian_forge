import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def _tree2etree(self, parent):
    from nltk.tree import Tree
    root = Element(parent.label())
    for child in parent:
        if isinstance(child, Tree):
            root.append(self._tree2etree(child))
        else:
            text, tag = child
            e = SubElement(root, tag)
            e.text = text
    return root