from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class DirectoryNotExistError(ArtifactRegistryError):
    """Raised when a directory does not exist."""