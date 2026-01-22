from __future__ import annotations
import sys
from queue import Empty, Queue
from kombu.exceptions import reraise
from kombu.log import get_logger
from kombu.utils.objects import cached_property
from . import virtual
@pyro.expose
@pyro.behavior(instance_mode='single')
class KombuBroker:
    """Kombu Broker used by the Pyro transport.

        You have to run this as a separate (Pyro) service.
        """

    def __init__(self):
        self.queues = {}

    def get_queue_names(self):
        return list(self.queues)

    def new_queue(self, queue):
        if queue in self.queues:
            return
        self.queues[queue] = Queue()

    def has_queue(self, queue):
        return queue in self.queues

    def get(self, queue):
        return self.queues[queue].get(block=False)

    def put(self, queue, message):
        self.queues[queue].put(message)

    def size(self, queue):
        return self.queues[queue].qsize()

    def delete(self, queue):
        del self.queues[queue]

    def purge(self, queue):
        while True:
            try:
                self.queues[queue].get(blocking=False)
            except Empty:
                break