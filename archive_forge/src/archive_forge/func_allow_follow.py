import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def allow_follow(self, anchor):
    """
        Override to suppress rel="nofollow" on some anchors.
        """
    return False