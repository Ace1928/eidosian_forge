from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsCreateRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsCreateRequest object.

  Fields:
    parent: Required. The parent resource name where the scan is created,
      which should be a project resource name in the format
      'projects/{projectId}'.
    scanConfig: A ScanConfig resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    scanConfig = _messages.MessageField('ScanConfig', 2)