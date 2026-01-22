import collections
import email.generator as generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import email.parser as email_parser
import itertools
import time
import uuid
import six
from six.moves import http_client
from six.moves import urllib_parse
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class RequestResponseAndHandler(collections.namedtuple('RequestResponseAndHandler', ['request', 'response', 'handler'])):
    """Container for data related to completing an HTTP request.

    This contains an HTTP request, its response, and a callback for handling
    the response from the server.

    Attributes:
      request: An http_wrapper.Request object representing the HTTP request.
      response: The http_wrapper.Response object returned from the server.
      handler: A callback function accepting two arguments, response
        and exception. Response is an http_wrapper.Response object, and
        exception is an apiclient.errors.HttpError object if an error
        occurred, or otherwise None.
    """