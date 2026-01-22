import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def __unescape(self, m):
    dval, hval, name = m.groups()
    if dval:
        codepoint = int(dval)
    elif hval:
        codepoint = int(hval, 16)
    else:
        codepoint = self.name2codepoint.get(name, 65533)
    if codepoint < 128:
        return chr(codepoint)
    return chr(codepoint)