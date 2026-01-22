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
def _SerializeRequest(self, request):
    """Convert a http_wrapper.Request object into a string.

        Args:
          request: A http_wrapper.Request to serialize.

        Returns:
          The request as a string in application/http format.
        """
    parsed = urllib_parse.urlsplit(request.url)
    request_line = urllib_parse.urlunsplit(('', '', parsed.path, parsed.query, ''))
    if not isinstance(request_line, six.text_type):
        request_line = request_line.decode('utf-8')
    status_line = u' '.join((request.http_method, request_line, u'HTTP/1.1\n'))
    major, minor = request.headers.get('content-type', 'application/json').split('/')
    msg = mime_nonmultipart.MIMENonMultipart(major, minor)
    for key, value in request.headers.items():
        if key == 'content-type':
            continue
        msg[key] = value
    msg['Host'] = parsed.netloc
    msg.set_unixfrom(None)
    if request.body is not None:
        msg.set_payload(request.body)
    str_io = six.StringIO()
    gen = generator.Generator(str_io, maxheaderlen=0)
    gen.flatten(msg, unixfrom=False)
    body = str_io.getvalue()
    return status_line + body