import logging
import socket
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
Make an HTTP request using http.client.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping): Request headers.
            timeout (Optional(int)): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                socket global default timeout will be used.
            kwargs: Additional arguments passed throught to the underlying
                :meth:`~http.client.HTTPConnection.request` method.

        Returns:
            Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        