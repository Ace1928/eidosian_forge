import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def _get_message_task_id(self, message):
    try:
        return message.properties['correlation_id']
    except (AttributeError, KeyError):
        return message.payload['task_id']