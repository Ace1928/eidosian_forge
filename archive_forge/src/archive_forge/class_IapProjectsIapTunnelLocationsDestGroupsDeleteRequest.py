from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsIapTunnelLocationsDestGroupsDeleteRequest(_messages.Message):
    """A IapProjectsIapTunnelLocationsDestGroupsDeleteRequest object.

  Fields:
    name: Required. Name of the TunnelDestGroup to delete. In the following
      format: `projects/{project_number/id}/iap_tunnel/locations/{location}/de
      stGroups/{dest_group}`.
  """
    name = _messages.StringField(1, required=True)