from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersPatchRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersPatchRequest
  object.

  Fields:
    name: Required. Resource name for the `ServicePerimeter`. Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`.
      The `service_perimeter` component must begin with a letter, followed by
      alphanumeric characters or `_`. After you create a `ServicePerimeter`,
      you cannot change its `name`.
    servicePerimeter: A ServicePerimeter resource to be passed as the request
      body.
    updateMask: Required. Mask to control which fields get updated. Must be
      non-empty.
  """
    name = _messages.StringField(1, required=True)
    servicePerimeter = _messages.MessageField('ServicePerimeter', 2)
    updateMask = _messages.StringField(3)