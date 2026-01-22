from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeystoresCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeystoresCreateRequest object.

  Fields:
    googleCloudApigeeV1Keystore: A GoogleCloudApigeeV1Keystore resource to be
      passed as the request body.
    name: Optional. Name of the keystore. Overrides the value in Keystore.
    parent: Required. Name of the environment in which to create the keystore.
      Use the following format in your request:
      `organizations/{org}/environments/{env}`
  """
    googleCloudApigeeV1Keystore = _messages.MessageField('GoogleCloudApigeeV1Keystore', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)