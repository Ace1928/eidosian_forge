import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def xml_escape(string):
    return re.sub('([&<"\\\'>])', lambda m: xml_escapes[m.group()], string)