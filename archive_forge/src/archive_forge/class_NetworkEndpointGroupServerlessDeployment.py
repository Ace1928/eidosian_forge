from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupServerlessDeployment(_messages.Message):
    """Configuration for a serverless network endpoint group (NEG). The
  platform must be provided. Note: The target backend service must be in the
  same project and located in the same region as the Serverless NEG.

  Fields:
    platform: The platform of the backend target(s) of this NEG. The only
      supported value is API Gateway: apigateway.googleapis.com.
    resource: The user-defined name of the workload/instance. This value must
      be provided explicitly or in the urlMask. The resource identified by
      this value is platform-specific and is as follows: 1. API Gateway: The
      gateway ID 2. App Engine: The service name 3. Cloud Functions: The
      function name 4. Cloud Run: The service name
    urlMask: An URL mask is one of the main components of the Cloud Function.
      A template to parse platform-specific fields from a request URL. URL
      mask allows for routing to multiple resources on the same serverless
      platform without having to create multiple Network Endpoint Groups and
      backend resources. The fields parsed by this template are platform-
      specific and are as follows: 1. API Gateway: The gateway ID 2. App
      Engine: The service and version 3. Cloud Functions: The function name 4.
      Cloud Run: The service and tag
    version: The optional resource version. The version identified by this
      value is platform-specific and is follows: 1. API Gateway: Unused 2. App
      Engine: The service version 3. Cloud Functions: Unused 4. Cloud Run: The
      service tag
  """
    platform = _messages.StringField(1)
    resource = _messages.StringField(2)
    urlMask = _messages.StringField(3)
    version = _messages.StringField(4)