from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsDeleteRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsDeleteRequest object.

  Fields:
    name: Required. The resource name of the ScanConfig to be deleted. The
      name follows the format of
      'projects/{projectId}/scanConfigs/{scanConfigId}'.
  """
    name = _messages.StringField(1, required=True)