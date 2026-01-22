import argparse
import json
import logging
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ray.autoscaler._private.local.node_provider import LocalNodeProvider
def _do_header(self, response_code=200, headers=None):
    """Sends the header portion of the HTTP response.

            Args:
                response_code: Standard HTTP response code
                headers (list[tuples]): Standard HTTP response headers
            """
    if headers is None:
        headers = [('Content-type', 'application/json')]
    self.send_response(response_code)
    for key, value in headers:
        self.send_header(key, value)
    self.end_headers()