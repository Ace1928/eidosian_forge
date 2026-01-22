from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DockerRepositoryConfig(_messages.Message):
    """DockerRepositoryConfig is docker related repository details. Provides
  additional configuration details for repositories of the docker format type.

  Fields:
    immutableTags: The repository which enabled this flag prevents all tags
      from being modified, moved or deleted. This does not prevent tags from
      being created.
  """
    immutableTags = _messages.BooleanField(1)