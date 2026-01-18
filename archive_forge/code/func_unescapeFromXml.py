from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def unescapeFromXml(text):
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&apos;', "'")
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    return text