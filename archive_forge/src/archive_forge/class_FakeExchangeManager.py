import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
class FakeExchangeManager(object):
    _exchanges_lock = threading.Lock()
    _exchanges = {}

    def __init__(self, default_exchange):
        self._default_exchange = default_exchange

    def get_exchange(self, name):
        if name is None:
            name = self._default_exchange
        with self._exchanges_lock:
            return self._exchanges.setdefault(name, FakeExchange(name))

    @classmethod
    def cleanup(cls):
        cls._exchanges.clear()