from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupAppEngine(_messages.Message):
    """Configuration for an App Engine network endpoint group (NEG). The
  service is optional, may be provided explicitly or in the URL mask. The
  version is optional and can only be provided explicitly or in the URL mask
  when service is present. Note: App Engine service must be in the same
  project and located in the same region as the Serverless NEG.

  Fields:
    service: Optional serving service. The service name is case-sensitive and
      must be 1-63 characters long. Example value: default, my-service.
    urlMask: An URL mask is one of the main components of the Cloud Function.
      A template to parse service and version fields from a request URL. URL
      mask allows for routing to multiple App Engine services without having
      to create multiple Network Endpoint Groups and backend services. For
      example, the request URLs foo1-dot-appname.appspot.com/v1 and foo1-dot-
      appname.appspot.com/v2 can be backed by the same Serverless NEG with URL
      mask <service>-dot-appname.appspot.com/<version>. The URL mask will
      parse them to { service = "foo1", version = "v1" } and { service =
      "foo1", version = "v2" } respectively.
    version: Optional serving version. The version name is case-sensitive and
      must be 1-100 characters long. Example value: v1, v2.
  """
    service = _messages.StringField(1)
    urlMask = _messages.StringField(2)
    version = _messages.StringField(3)