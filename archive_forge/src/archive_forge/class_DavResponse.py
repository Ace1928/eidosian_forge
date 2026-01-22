import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavResponse(urllib.Response):
    """Custom HTTPResponse.

    DAV have some reponses for which the body is of no interest.
    """
    _body_ignored_responses = urllib.Response._body_ignored_responses + [201, 405, 409, 412]

    def begin(self):
        """Begin to read the response from the server.

        httplib incorrectly close the connection far too easily. Let's try to
        workaround that (as urllib does, but for more cases...).
        """
        urllib.Response.begin(self)
        if self.status in (201, 204):
            self.will_close = False