import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _xml_escape_attr(attr, skip_single_quote=True):
    """Escape the given string for use in an HTML/XML tag attribute.

    By default this doesn't bother with escaping `'` to `&#39;`, presuming that
    the tag attribute is surrounded by double quotes.
    """
    escaped = _AMPERSAND_RE.sub('&amp;', attr)
    escaped = attr.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
    if not skip_single_quote:
        escaped = escaped.replace("'", '&#39;')
    return escaped