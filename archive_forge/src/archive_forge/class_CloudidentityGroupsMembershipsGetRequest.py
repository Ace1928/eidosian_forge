from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsGetRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsGetRequest object.

  Fields:
    name: Required. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Membership` to retrieve. Must be of the form
      `groups/{group}/memberships/{membership}`.
  """
    name = _messages.StringField(1, required=True)