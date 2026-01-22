from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsSecurityGatewaysGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsSecurityGatewaysGetRequest object.

  Fields:
    name: Required. The resource name of the PartnerTenant using the form: `pr
      ojects/{project_id}/locations/{location_id}/securityGateway/{security_ga
      teway_id}`
  """
    name = _messages.StringField(1, required=True)