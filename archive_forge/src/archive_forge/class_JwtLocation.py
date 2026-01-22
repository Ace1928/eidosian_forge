from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JwtLocation(_messages.Message):
    """Specifies a location to extract JWT from an API request.

  Fields:
    cookie: Specifies cookie name to extract JWT token.
    header: Specifies HTTP header name to extract JWT token.
    query: Specifies URL query parameter name to extract JWT token.
    valuePrefix: The value prefix. The value format is "value_prefix{token}"
      Only applies to "in" header type. Must be empty for "in" query type. If
      not empty, the header value has to match (case sensitive) this prefix.
      If not matched, JWT will not be extracted. If matched, JWT will be
      extracted after the prefix is removed. For example, for "Authorization:
      Bearer {JWT}", value_prefix="Bearer " with a space at the end.
  """
    cookie = _messages.StringField(1)
    header = _messages.StringField(2)
    query = _messages.StringField(3)
    valuePrefix = _messages.StringField(4)