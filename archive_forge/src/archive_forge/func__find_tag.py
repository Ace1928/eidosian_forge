import sys
import string
from html5lib import HTMLParser as _HTMLParser
from html5lib.treebuilders.etree_lxml import TreeBuilder
from lxml import etree
from lxml.html import Element, XHTML_NAMESPACE, _contains_block_level_tag
def _find_tag(tree, tag):
    elem = tree.find(tag)
    if elem is not None:
        return elem
    return tree.find('{%s}%s' % (XHTML_NAMESPACE, tag))