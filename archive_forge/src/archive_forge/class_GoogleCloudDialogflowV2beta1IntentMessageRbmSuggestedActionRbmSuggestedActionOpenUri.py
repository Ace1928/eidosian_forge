from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedActionRbmSuggestedActionOpenUri(_messages.Message):
    """Opens the user's default web browser app to the specified uri If the
  user has an app installed that is registered as the default handler for the
  URL, then this app will be opened instead, and its icon will be used in the
  suggested action UI.

  Fields:
    uri: Required. The uri to open on the user device
  """
    uri = _messages.StringField(1)