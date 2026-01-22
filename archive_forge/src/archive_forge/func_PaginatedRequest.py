from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import re
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_creds as v2_2_creds
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def PaginatedRequest(self, url, accepted_codes=None, method=None, body=None, content_type=None):
    """Wrapper around Request that follows Link headers if they exist.

    Args:
      url: the URL to which to talk
      accepted_codes: the list of acceptable http status codes
      method: the HTTP method to use (defaults to GET/PUT depending on
              whether body is provided)
      body: the body to pass into the PUT request (or None for GET)
      content_type: the mime-type of the request (or None for JSON)

    Yields:
      The return value of calling Request for each page of results.
    """
    next_page = url
    while next_page:
        resp, content = self.Request(next_page, accepted_codes, method, body, content_type)
        yield (resp, content)
        next_page = ParseNextLinkHeader(resp)