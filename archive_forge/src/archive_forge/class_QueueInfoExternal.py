from __future__ import absolute_import
from __future__ import unicode_literals
import os
class QueueInfoExternal(validation.Validated):
    """Describes all of the queue entries for an application."""
    ATTRIBUTES = {appinfo.APPLICATION: validation.Optional(appinfo.APPLICATION_RE_STRING), TOTAL_STORAGE_LIMIT: validation.Optional(_TOTAL_STORAGE_LIMIT_REGEX), RESUME_PAUSED_QUEUES: validation.Optional(_RESUME_PAUSED_QUEUES), QUEUE: validation.Optional(validation.Repeated(QueueEntry))}