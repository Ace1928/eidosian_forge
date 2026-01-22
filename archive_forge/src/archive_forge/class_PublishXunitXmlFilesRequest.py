from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublishXunitXmlFilesRequest(_messages.Message):
    """Request message for StepService.PublishXunitXmlFiles.

  Fields:
    xunitXmlFiles: URI of the Xunit XML files to publish. The maximum size of
      the file this reference is pointing to is 50MB. Required.
  """
    xunitXmlFiles = _messages.MessageField('FileReference', 1, repeated=True)