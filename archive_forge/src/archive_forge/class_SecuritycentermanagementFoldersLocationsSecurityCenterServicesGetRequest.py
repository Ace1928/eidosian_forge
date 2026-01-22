from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementFoldersLocationsSecurityCenterServicesGetRequest(_messages.Message):
    """A
  SecuritycentermanagementFoldersLocationsSecurityCenterServicesGetRequest
  object.

  Fields:
    name: Required. The Security Center service to retrieve. Formats: * organi
      zations/{organization}/locations/{location}/securityCenterServices/{serv
      ice} *
      folders/{folder}/locations/{location}/securityCenterServices/{service} *
      projects/{project}/locations/{location}/securityCenterServices/{service}
  """
    name = _messages.StringField(1, required=True)