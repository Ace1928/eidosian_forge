import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def url_escape(string):
    string = string.encode('utf8')
    return quote_plus(string)