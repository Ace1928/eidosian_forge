from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsKeysApiproductsUpdateDeveloperAppKeyApiProductRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsKeysApiproductsUpdateDeveloperAppKeyA
  piProductRequest object.

  Fields:
    action: Approve or revoke the consumer key by setting this value to
      `approve` or `revoke`, respectively.
    name: Name of the API product in the developer app key in the following
      format: `organizations/{org}/developers/{developer_email}/apps/{app}/key
      s/{key}/apiproducts/{apiproduct}`
  """
    action = _messages.StringField(1)
    name = _messages.StringField(2, required=True)