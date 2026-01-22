from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpHeaderMatch(_messages.Message):
    """Specification of HTTP header match attributes.

  Fields:
    headerName: Required. The name of the HTTP header to match. For matching
      against the HTTP request's authority, use a headerMatch with the header
      name ":authority". For matching a request's method, use the headerName
      ":method".
    regexMatch: Required. The value of the header must match the regular
      expression specified in regexMatch. For regular expression grammar,
      please see: en.cppreference.com/w/cpp/regex/ecmascript For matching
      against a port specified in the HTTP request, use a headerMatch with
      headerName set to Host and a regular expression that satisfies the
      RFC2616 Host header's port specifier.
  """
    headerName = _messages.StringField(1)
    regexMatch = _messages.StringField(2)