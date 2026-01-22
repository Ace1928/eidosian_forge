from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFilesResponse(_messages.Message):
    """The response from listing files.

  Fields:
    files: The files returned.
    nextPageToken: The token to retrieve the next page of files, or empty if
      there are no more files to return.
  """
    files = _messages.MessageField('GoogleDevtoolsArtifactregistryV1File', 1, repeated=True)
    nextPageToken = _messages.StringField(2)