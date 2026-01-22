from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GrpcOperationGroup(_messages.Message):
    """List of gRPC operation configuration details associated with Apigee API
  proxies.

  Fields:
    operationConfigs: Required. List of operation configurations for either
      Apigee API proxies that are associated with this API product.
  """
    operationConfigs = _messages.MessageField('GoogleCloudApigeeV1GrpcOperationConfig', 1, repeated=True)