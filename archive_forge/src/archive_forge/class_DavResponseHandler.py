import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavResponseHandler(xml.sax.handler.ContentHandler):
    """Handle a multi-status DAV response."""

    def __init__(self):
        self.url = None
        self.elt_stack = None
        self.chars = None
        self.chars_wanted = False
        self.expected_content_handled = False

    def set_url(self, url):
        """Set the url used for error reporting when handling a response."""
        self.url = url

    def startDocument(self):
        self.elt_stack = []
        self.chars = None
        self.expected_content_handled = False

    def endDocument(self):
        self._validate_handling()
        if not self.expected_content_handled:
            raise errors.InvalidHttpResponse(self.url, msg='Unknown xml response')

    def startElement(self, name, attrs):
        self.elt_stack.append(self._strip_ns(name))
        if self.chars_wanted:
            self.chars = ''
        else:
            self.chars = None

    def endElement(self, name):
        self.chars = None
        self.chars_wanted = False
        self.elt_stack.pop()

    def characters(self, chrs):
        if self.chars_wanted:
            self.chars += chrs

    def _current_element(self):
        return self.elt_stack[-1]

    def _strip_ns(self, name):
        """Strip the leading namespace from name.

        We don't have namespaces clashes in our context, stripping it makes the
        code simpler.
        """
        where = name.find(':')
        if where == -1:
            return name
        else:
            return name[where + 1:]