from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class HttpHeadersDict(validation.ValidatedDict):
    """A dict that limits keys and values to what `http_headers` allows.

  `http_headers` is an static handler key; it applies to handlers with
  `static_dir` or `static_files` keys. The following code is an example of how
  `http_headers` is used::

      handlers:
      - url: /static
        static_dir: static
        http_headers:
          X-Foo-Header: foo value
          X-Bar-Header: bar value

  """
    DISALLOWED_HEADERS = frozenset(['content-encoding', 'content-length', 'date', 'server'])
    MAX_HEADER_LENGTH = 500
    MAX_HEADER_VALUE_LENGTHS = {'content-security-policy': _MAX_HEADER_SIZE_FOR_EXEMPTED_HEADERS, 'x-content-security-policy': _MAX_HEADER_SIZE_FOR_EXEMPTED_HEADERS, 'x-webkit-csp': _MAX_HEADER_SIZE_FOR_EXEMPTED_HEADERS, 'content-security-policy-report-only': _MAX_HEADER_SIZE_FOR_EXEMPTED_HEADERS, 'set-cookie': _MAX_COOKIE_LENGTH, 'set-cookie2': _MAX_COOKIE_LENGTH, 'location': _MAX_URL_LENGTH}
    MAX_LEN = 500

    class KeyValidator(validation.Validator):
        """Ensures that keys in `HttpHeadersDict` are valid.

    `HttpHeadersDict` contains a list of headers. An instance is used as
    `HttpHeadersDict`'s `KEY_VALIDATOR`.
    """

        def Validate(self, name, unused_key=None):
            """Returns an argument, or raises an exception if the argument is invalid.

      HTTP header names are defined by `RFC 2616, section 4.2`_.

      Args:
        name: HTTP header field value.
        unused_key: Unused.

      Returns:
        name argument, unchanged.

      Raises:
        appinfo_errors.InvalidHttpHeaderName: An argument cannot be used as an
            HTTP header name.

      .. _RFC 2616, section 4.2:
         https://www.ietf.org/rfc/rfc2616.txt
      """
            original_name = name
            if isinstance(name, six_subset.string_types):
                name = EnsureAsciiBytes(name, appinfo_errors.InvalidHttpHeaderName('HTTP header values must not contain non-ASCII data'))
            name = name.lower().decode('ascii')
            if not _HTTP_TOKEN_RE.match(name):
                raise appinfo_errors.InvalidHttpHeaderName('An HTTP header must be a non-empty RFC 2616 token.')
            if name in _HTTP_REQUEST_HEADERS:
                raise appinfo_errors.InvalidHttpHeaderName('%r can only be used in HTTP requests, not responses.' % original_name)
            if name.startswith('x-appengine'):
                raise appinfo_errors.InvalidHttpHeaderName('HTTP header names that begin with X-Appengine are reserved.')
            if wsgiref.util.is_hop_by_hop(name):
                raise appinfo_errors.InvalidHttpHeaderName('Only use end-to-end headers may be used. See RFC 2616 section 13.5.1.')
            if name in HttpHeadersDict.DISALLOWED_HEADERS:
                raise appinfo_errors.InvalidHttpHeaderName('%s is a disallowed header.' % name)
            return original_name

    class ValueValidator(validation.Validator):
        """Ensures that values in `HttpHeadersDict` are valid.

    An instance is used as `HttpHeadersDict`'s `VALUE_VALIDATOR`.
    """

        def Validate(self, value, key=None):
            """Returns a value, or raises an exception if the value is invalid.

      According to `RFC 2616 section 4.2`_ header field values must consist "of
      either *TEXT or combinations of token, separators, and quoted-string"::

          TEXT = <any OCTET except CTLs, but including LWS>

      Args:
        value: HTTP header field value.
        key: HTTP header field name.

      Returns:
        A value argument.

      Raises:
        appinfo_errors.InvalidHttpHeaderValue: An argument cannot be used as an
            HTTP header value.

      .. _RFC 2616, section 4.2:
         https://www.ietf.org/rfc/rfc2616.txt
      """
            error = appinfo_errors.InvalidHttpHeaderValue('HTTP header values must not contain non-ASCII data')
            if isinstance(value, six_subset.string_types):
                b_value = EnsureAsciiBytes(value, error)
            else:
                b_value = EnsureAsciiBytes('%s' % value, error)
            key = key.lower()
            printable = set(string.printable[:-5].encode('ascii'))
            if not all((b in printable for b in b_value)):
                raise appinfo_errors.InvalidHttpHeaderValue('HTTP header field values must consist of printable characters.')
            HttpHeadersDict.ValueValidator.AssertHeaderNotTooLong(key, value)
            return value

        @staticmethod
        def AssertHeaderNotTooLong(name, value):
            header_length = len(('%s: %s\r\n' % (name, value)).encode('ascii'))
            if header_length >= HttpHeadersDict.MAX_HEADER_LENGTH:
                try:
                    max_len = HttpHeadersDict.MAX_HEADER_VALUE_LENGTHS[name]
                except KeyError:
                    raise appinfo_errors.InvalidHttpHeaderValue('HTTP header (name + value) is too long.')
                if len(value) > max_len:
                    insert = (name, len(value), max_len)
                    raise appinfo_errors.InvalidHttpHeaderValue('%r header value has length %d, which exceed the maximum allowed, %d.' % insert)
    KEY_VALIDATOR = KeyValidator()
    VALUE_VALIDATOR = ValueValidator()

    def Get(self, header_name):
        """Gets a header value.

    Args:
      header_name: HTTP header name to look for.

    Returns:
      A header value that corresponds to `header_name`. If more than one such
      value is in `self`, one of the values is selected arbitrarily and
      returned. The selection is not deterministic.
    """
        for name in self:
            if name.lower() == header_name.lower():
                return self[name]

    def __setitem__(self, key, value):
        is_addition = self.Get(key) is None
        if is_addition and len(self) >= self.MAX_LEN:
            raise appinfo_errors.TooManyHttpHeaders('Tried to add another header when the current set of HTTP headers already has the maximum allowed number of headers, %d.' % HttpHeadersDict.MAX_LEN)
        super(HttpHeadersDict, self).__setitem__(key, value)