from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CorsPolicy(_messages.Message):
    """The specification for allowing client-side cross-origin requests. For
  more information about the W3C recommendation for cross-origin resource
  sharing (CORS), see Fetch API Living Standard.

  Fields:
    allowCredentials: In response to a preflight request, setting this to true
      indicates that the actual request can include user credentials. This
      field translates to the Access-Control-Allow-Credentials header. Default
      is false.
    allowHeaders: Specifies the content for the Access-Control-Allow-Headers
      header.
    allowMethods: Specifies the content for the Access-Control-Allow-Methods
      header.
    allowOriginRegexes: Specifies a regular expression that matches allowed
      origins. For more information, see regular expression syntax . An origin
      is allowed if it matches either an item in allowOrigins or an item in
      allowOriginRegexes. Regular expressions can only be used when the
      loadBalancingScheme is set to INTERNAL_SELF_MANAGED.
    allowOrigins: Specifies the list of origins that is allowed to do CORS
      requests. An origin is allowed if it matches either an item in
      allowOrigins or an item in allowOriginRegexes.
    disabled: If true, disables the CORS policy. The default value is false,
      which indicates that the CORS policy is in effect.
    exposeHeaders: Specifies the content for the Access-Control-Expose-Headers
      header.
    maxAge: Specifies how long results of a preflight request can be cached in
      seconds. This field translates to the Access-Control-Max-Age header.
  """
    allowCredentials = _messages.BooleanField(1)
    allowHeaders = _messages.StringField(2, repeated=True)
    allowMethods = _messages.StringField(3, repeated=True)
    allowOriginRegexes = _messages.StringField(4, repeated=True)
    allowOrigins = _messages.StringField(5, repeated=True)
    disabled = _messages.BooleanField(6)
    exposeHeaders = _messages.StringField(7, repeated=True)
    maxAge = _messages.IntegerField(8, variant=_messages.Variant.INT32)