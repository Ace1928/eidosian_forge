from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1KeystoreConfig(_messages.Message):
    """A GoogleCloudApigeeV1KeystoreConfig object.

  Fields:
    aliases: Aliases in the keystore.
    name: Resource name in the following format:
      `organizations/{org}/environments/{env}/keystores/{keystore}`
  """
    aliases = _messages.MessageField('GoogleCloudApigeeV1AliasRevisionConfig', 1, repeated=True)
    name = _messages.StringField(2)