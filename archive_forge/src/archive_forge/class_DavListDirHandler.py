import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavListDirHandler(DavStatHandler):
    """Handle a PROPPFIND depth 1 DAV response for a directory."""

    def __init__(self):
        DavStatHandler.__init__(self)
        self.dir_content = None

    def _validate_handling(self):
        if self.dir_content is not None:
            self.expected_content_handled = True

    def _make_response_tuple(self):
        if self.executable == 'T':
            is_exec = True
        else:
            is_exec = False
        return (self.href, self.is_dir, self.length, is_exec)

    def _response_handled(self):
        """A response element inside a multistatus have been parsed."""
        if self.dir_content is None:
            self.dir_content = []
        self.dir_content.append(self._make_response_tuple())
        self._init_response_attrs()

    def _additional_response_starting(self, name):
        """A additional response element inside a multistatus begins."""
        pass