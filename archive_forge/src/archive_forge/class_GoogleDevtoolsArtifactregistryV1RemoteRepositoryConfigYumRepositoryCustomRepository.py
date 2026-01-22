from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryCustomRepository(_messages.Message):
    """Customer-specified publicly available remote repository.

  Fields:
    uri: An http/https uri reference to the upstream remote repository, for
      ex: "https://my.yum.registry/".
  """
    uri = _messages.StringField(1)