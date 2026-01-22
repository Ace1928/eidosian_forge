from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1UpdateAppGroupAppKeyRequest(_messages.Message):
    """Request for UpdateAppGroupAppKey

  Fields:
    action: Approve or revoke the consumer key by setting this value to
      `approve` or `revoke` respectively. The `Content-Type` header, if set,
      must be set to `application/octet-stream`, with empty body.
    apiProducts: The list of API products that will be associated with the
      credential. This list will be appended to the existing list of
      associated API Products for this App Key. Duplicates will be ignored.
    appGroupAppKey: The new AppGroupKey to be amended. Note that the status
      can be updated only via action.
  """
    action = _messages.StringField(1)
    apiProducts = _messages.StringField(2, repeated=True)
    appGroupAppKey = _messages.MessageField('GoogleCloudApigeeV1AppGroupAppKey', 3)