from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNetworksPatchRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNetworksPatchRequest object.

  Fields:
    name: Output only. The resource name of this `Network`. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. Format:
      `projects/{project}/locations/{location}/networks/{network}`
    network: A Network resource to be passed as the request body.
    updateMask: The list of fields to update. The only currently supported
      fields are: `labels`, `reservations`, `vrf.vlan_attachments`
  """
    name = _messages.StringField(1, required=True)
    network = _messages.MessageField('Network', 2)
    updateMask = _messages.StringField(3)