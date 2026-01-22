from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def GetFieldAndLogUnreachableInstancePartitions(message, attribute):
    """Response callback to log unreachable while generating fields of the message."""
    warning_text = 'The following instance partitions were unreachable: {}.'
    if hasattr(message, 'unreachable') and message.unreachable:
        log.warning(warning_text.format(', '.join(message.unreachable)))
    elif hasattr(message, 'unreachableInstancePartitions') and message.unreachableInstancePartitions:
        log.warning(warning_text.format(', '.join(message.unreachableInstancePartitions)))
    return getattr(message, attribute)