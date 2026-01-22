from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeystoresGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeystoresGetRequest object.

  Fields:
    name: Required. Name of the keystore. Use the following format in your
      request: `organizations/{org}/environments/{env}/keystores/{keystore}`.
  """
    name = _messages.StringField(1, required=True)