from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceReference(_messages.Message):
    """A reference to a particular snapshot of the source tree used to build
  and deploy an application.

  Fields:
    repository: Optional. A URI string identifying the repository. Example:
      "https://github.com/GoogleCloudPlatform/kubernetes.git"
    revisionId: The canonical and persistent identifier of the deployed
      revision. Example (git): "0035781c50ec7aa23385dc841529ce8a4b70db1b"
  """
    repository = _messages.StringField(1)
    revisionId = _messages.StringField(2)