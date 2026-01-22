from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsTerraformVersionsGetRequest(_messages.Message):
    """A ConfigProjectsLocationsTerraformVersionsGetRequest object.

  Fields:
    name: Required. The name of the TerraformVersion. Format: 'projects/{proje
      ct_id}/locations/{location}/terraformVersions/{terraform_version}'
  """
    name = _messages.StringField(1, required=True)