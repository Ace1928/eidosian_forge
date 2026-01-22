from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersGetMonetizationConfigRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersGetMonetizationConfigRequest object.

  Fields:
    name: Required. Monetization configuration for the developer. Use the
      following structure in your request:
      `organizations/{org}/developers/{developer}/monetizationConfig`
  """
    name = _messages.StringField(1, required=True)