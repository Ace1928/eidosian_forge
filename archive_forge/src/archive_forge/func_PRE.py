from html import escape
from io import StringIO
from incremental import Version
from twisted.python import log
from twisted.python.deprecate import deprecated
@deprecated(Version('Twisted', 15, 3, 0), replacement='twisted.web.template')
def PRE(text):
    """Wrap <pre> tags around some text and HTML-escape it."""
    return '<pre>' + escape(text) + '</pre>'