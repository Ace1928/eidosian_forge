from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceTerraformInfo(_messages.Message):
    """Terraform info of a Resource.

  Fields:
    address: TF resource address that uniquely identifies this resource within
      this deployment.
    id: ID attribute of the TF resource
    type: TF resource type
  """
    address = _messages.StringField(1)
    id = _messages.StringField(2)
    type = _messages.StringField(3)