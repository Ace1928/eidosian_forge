import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def _slurp_from_queue(self, task_id, accept, limit=1000, no_ack=False):
    with self.app.pool.acquire_channel(block=True) as (_, channel):
        binding = self._create_binding(task_id)(channel)
        binding.declare()
        for _ in range(limit):
            msg = binding.get(accept=accept, no_ack=no_ack)
            if not msg:
                break
            yield msg
        else:
            raise self.BacklogLimitExceeded(task_id)