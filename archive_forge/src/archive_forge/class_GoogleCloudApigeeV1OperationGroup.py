from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OperationGroup(_messages.Message):
    """List of operation configuration details associated with Apigee API
  proxies or remote services. Remote services are non-Apigee proxies, such as
  Istio-Envoy.

  Fields:
    operationConfigType: Flag that specifes whether the configuration is for
      Apigee API proxy or a remote service. Valid values include `proxy` or
      `remoteservice`. Defaults to `proxy`. Set to `proxy` when Apigee API
      proxies are associated with the API product. Set to `remoteservice` when
      non-Apigee proxies like Istio-Envoy are associated with the API product.
    operationConfigs: Required. List of operation configurations for either
      Apigee API proxies or other remote services that are associated with
      this API product.
  """
    operationConfigType = _messages.StringField(1)
    operationConfigs = _messages.MessageField('GoogleCloudApigeeV1OperationConfig', 2, repeated=True)