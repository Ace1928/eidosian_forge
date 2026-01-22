from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateVersionMetadataV1Beta(_messages.Message):
    """Metadata for the given google.longrunning.Operation during a
  google.appengine.v1beta.CreateVersionRequest.

  Fields:
    cloudBuildId: The Cloud Build ID if one was created as part of the version
      create. @OutputOnly
  """
    cloudBuildId = _messages.StringField(1)