import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavStatHandler(DavResponseHandler):
    """Handle a PROPPFIND DAV response for a file or directory.

    The expected content is:
    - a multi-status element containing
      - a single response element containing
        - a href element
        - a propstat element containing
          - a status element (ignored)
          - a prop element containing at least (other are ignored)
            - a getcontentlength element (for files only)
            - an executable element (for files only)
            - a resourcetype element containing
              - a collection element (for directories only)
    """

    def __init__(self):
        DavResponseHandler.__init__(self)
        self._response_seen = False
        self._init_response_attrs()

    def _init_response_attrs(self):
        self.href = None
        self.length = -1
        self.executable = None
        self.is_dir = False

    def _validate_handling(self):
        if self.href is not None:
            self.expected_content_handled = True

    def startElement(self, name, attrs):
        sname = self._strip_ns(name)
        self.chars_wanted = sname in ('href', 'getcontentlength', 'executable')
        DavResponseHandler.startElement(self, name, attrs)

    def endElement(self, name):
        if self._response_seen:
            self._additional_response_starting(name)
        if self._href_end():
            self.href = self.chars
        elif self._getcontentlength_end():
            self.length = int(self.chars)
        elif self._executable_end():
            self.executable = self.chars
        elif self._collection_end():
            self.is_dir = True
        if self._strip_ns(name) == 'response':
            self._response_seen = True
            self._response_handled()
        DavResponseHandler.endElement(self, name)

    def _response_handled(self):
        """A response element inside a multistatus have been parsed."""
        pass

    def _additional_response_starting(self, name):
        """A additional response element inside a multistatus begins."""
        sname = self._strip_ns(name)
        if sname != 'multistatus':
            raise errors.InvalidHttpResponse(self.url, msg='Unexpected %s element' % name)

    def _href_end(self):
        stack = self.elt_stack
        return len(stack) == 3 and stack[0] == 'multistatus' and (stack[1] == 'response') and (stack[2] == 'href')

    def _getcontentlength_end(self):
        stack = self.elt_stack
        return len(stack) == 5 and stack[0] == 'multistatus' and (stack[1] == 'response') and (stack[2] == 'propstat') and (stack[3] == 'prop') and (stack[4] == 'getcontentlength')

    def _executable_end(self):
        stack = self.elt_stack
        return len(stack) == 5 and stack[0] == 'multistatus' and (stack[1] == 'response') and (stack[2] == 'propstat') and (stack[3] == 'prop') and (stack[4] == 'executable')

    def _collection_end(self):
        stack = self.elt_stack
        return len(stack) == 6 and stack[0] == 'multistatus' and (stack[1] == 'response') and (stack[2] == 'propstat') and (stack[3] == 'prop') and (stack[4] == 'resourcetype') and (stack[5] == 'collection')