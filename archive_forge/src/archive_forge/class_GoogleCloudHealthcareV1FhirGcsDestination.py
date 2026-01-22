from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1FhirGcsDestination(_messages.Message):
    """The configuration for exporting to Cloud Storage.

  Fields:
    uriPrefix: URI for a Cloud Storage directory where result files should be
      written, in the format of `gs://{bucket-id}/{path/to/destination/dir}`.
      If there is no trailing slash, the service appends one when composing
      the object path. The user is responsible for creating the Cloud Storage
      bucket referenced in `uri_prefix`.
  """
    uriPrefix = _messages.StringField(1)