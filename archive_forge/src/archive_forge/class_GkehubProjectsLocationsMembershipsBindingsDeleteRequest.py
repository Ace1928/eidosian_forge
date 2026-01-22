from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsBindingsDeleteRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsBindingsDeleteRequest object.

  Fields:
    name: Required. The MembershipBinding resource name in the format
      `projects/*/locations/*/memberships/*/bindings/*`.
  """
    name = _messages.StringField(1, required=True)