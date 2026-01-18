import re
from lxml import etree, html
def unescape_entity(m):
    try:
        return unichr(name2codepoint[m.group(1)])
    except KeyError:
        return m.group(0)