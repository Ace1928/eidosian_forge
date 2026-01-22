from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigDockerRepositoryCustomRepository(_messages.Message):
    """Customer-specified publicly available remote repository.

  Fields:
    uri: An http/https uri reference to the custom remote repository, for ex:
      "https://registry-1.docker.io".
  """
    uri = _messages.StringField(1)