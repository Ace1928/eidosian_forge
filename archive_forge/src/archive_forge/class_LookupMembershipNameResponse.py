from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupMembershipNameResponse(_messages.Message):
    """The response message for MembershipsService.LookupMembershipName.

  Fields:
    name: The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      looked-up `Membership`. Must be of the form
      `groups/{group}/memberships/{membership}`.
  """
    name = _messages.StringField(1)