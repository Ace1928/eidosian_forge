from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsFindingsGetRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsScanRunsFindingsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Finding to be returned. The name
      follows the format of 'projects/{projectId}/scanConfigs/{scanConfigId}/s
      canRuns/{scanRunId}/findings/{findingId}'.
  """
    name = _messages.StringField(1, required=True)