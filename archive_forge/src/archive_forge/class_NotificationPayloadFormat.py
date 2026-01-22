from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
class NotificationPayloadFormat(enum.Enum):
    """Used to format the body of notifications."""
    JSON = 'json'
    NONE = 'none'