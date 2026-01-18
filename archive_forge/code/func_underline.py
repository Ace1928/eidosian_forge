from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def underline(self, text, pad_char):
    text = (text or '').rstrip()
    return '%s\n%s\n\n' % (text, pad_char * len(text)) if text else ''