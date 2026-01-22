from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdvancedOptionsConfigJsonCustomConfig(_messages.Message):
    """A SecurityPolicyAdvancedOptionsConfigJsonCustomConfig object.

  Fields:
    contentTypes: A list of custom Content-Type header values to apply the
      JSON parsing. As per RFC 1341, a Content-Type header value has the
      following format: Content-Type := type "/" subtype *[";" parameter] When
      configuring a custom Content-Type header value, only the type/subtype
      needs to be specified, and the parameters should be excluded.
  """
    contentTypes = _messages.StringField(1, repeated=True)