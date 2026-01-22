from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IaC(_messages.Message):
    """Infrastrucutre as code representations.

  Fields:
    tfPlan: Optional. Terraform plan file in JSON format. See
      https://developer.hashicorp.com/terraform/internals/json-format for how
      to generate JSON representation of a terraform plan file.
  """
    tfPlan = _messages.BytesField(1)