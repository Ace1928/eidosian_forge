import re
from lxml import etree, html
def append_text(parent, text):
    if len(parent) == 0:
        parent.text = (parent.text or '') + text
    else:
        parent[-1].tail = (parent[-1].tail or '') + text