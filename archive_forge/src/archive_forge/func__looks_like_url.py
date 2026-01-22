import sys
import string
from html5lib import HTMLParser as _HTMLParser
from html5lib.treebuilders.etree_lxml import TreeBuilder
from lxml import etree
from lxml.html import Element, XHTML_NAMESPACE, _contains_block_level_tag
def _looks_like_url(str):
    scheme = urlparse(str)[0]
    if not scheme:
        return False
    elif sys.platform == 'win32' and scheme in string.ascii_letters and (len(scheme) == 1):
        return False
    else:
        return True