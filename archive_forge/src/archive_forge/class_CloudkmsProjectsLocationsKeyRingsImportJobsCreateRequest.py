from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsImportJobsCreateRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsImportJobsCreateRequest object.

  Fields:
    importJob: A ImportJob resource to be passed as the request body.
    importJobId: Required. It must be unique within a KeyRing and match the
      regular expression `[a-zA-Z0-9_-]{1,63}`
    parent: Required. The name of the KeyRing associated with the ImportJobs.
  """
    importJob = _messages.MessageField('ImportJob', 1)
    importJobId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)