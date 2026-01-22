from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfileLocation(_messages.Message):
    """The data that will be profiled.

  Fields:
    folderId: The ID of the Folder within an organization to scan.
    organizationId: The ID of an organization to scan.
  """
    folderId = _messages.IntegerField(1)
    organizationId = _messages.IntegerField(2)