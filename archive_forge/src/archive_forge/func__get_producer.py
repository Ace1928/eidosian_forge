from __future__ import annotations
from queue import Empty
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import str_to_bytes
from kombu.utils.json import dumps, loads
from kombu.log import get_logger
def _get_producer(self, queue):
    """Create/get a producer instance for the given topic/queue."""
    queue = self.sanitize_queue_name(queue)
    producer = self._kafka_producers.get(queue, None)
    if producer is None:
        producer = Producer({**self.common_config, **(self.options.get('kafka_producer_config') or {})})
        self._kafka_producers[queue] = producer
    return producer