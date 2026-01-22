from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageImport(_messages.Message):
    """ImageImport describes the configuration of the image import to run.

  Fields:
    cloudStorageUri: Immutable. The path to the Cloud Storage file from which
      the image should be imported.
    createTime: Output only. The time the image import was created.
    diskImageTargetDefaults: Immutable. Target details for importing a disk
      image, will be used by ImageImportJob.
    encryption: Immutable. The encryption details used by the image import
      process during the image adaptation for Compute Engine.
    name: Output only. The resource path of the ImageImport.
    recentImageImportJobs: Output only. The result of the most recent runs for
      this ImageImport. All jobs for this ImageImport can be listed via
      ListImageImportJobs.
  """
    cloudStorageUri = _messages.StringField(1)
    createTime = _messages.StringField(2)
    diskImageTargetDefaults = _messages.MessageField('DiskImageTargetDetails', 3)
    encryption = _messages.MessageField('Encryption', 4)
    name = _messages.StringField(5)
    recentImageImportJobs = _messages.MessageField('ImageImportJob', 6, repeated=True)