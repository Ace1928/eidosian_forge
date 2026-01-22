from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
class UploadStrategy(enum.Enum):
    """Enum class for specifying upload strategy."""
    SIMPLE = 'simple'
    RESUMABLE = 'resumable'
    STREAMING = 'streaming'