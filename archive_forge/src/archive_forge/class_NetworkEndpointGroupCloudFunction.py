from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupCloudFunction(_messages.Message):
    """Configuration for a Cloud Function network endpoint group (NEG). The
  function must be provided explicitly or in the URL mask. Note: Cloud
  Function must be in the same project and located in the same region as the
  Serverless NEG.

  Fields:
    function: A user-defined name of the Cloud Function. The function name is
      case-sensitive and must be 1-63 characters long. Example value: func1.
    urlMask: An URL mask is one of the main components of the Cloud Function.
      A template to parse function field from a request URL. URL mask allows
      for routing to multiple Cloud Functions without having to create
      multiple Network Endpoint Groups and backend services. For example,
      request URLs mydomain.com/function1 and mydomain.com/function2 can be
      backed by the same Serverless NEG with URL mask /<function>. The URL
      mask will parse them to { function = "function1" } and { function =
      "function2" } respectively.
  """
    function = _messages.StringField(1)
    urlMask = _messages.StringField(2)