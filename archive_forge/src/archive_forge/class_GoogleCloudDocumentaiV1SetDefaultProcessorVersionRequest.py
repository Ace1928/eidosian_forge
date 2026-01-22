from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1SetDefaultProcessorVersionRequest(_messages.Message):
    """Request message for the SetDefaultProcessorVersion method.

  Fields:
    defaultProcessorVersion: Required. The resource name of child
      ProcessorVersion to use as default. Format: `projects/{project}/location
      s/{location}/processors/{processor}/processorVersions/{version}`
  """
    defaultProcessorVersion = _messages.StringField(1)