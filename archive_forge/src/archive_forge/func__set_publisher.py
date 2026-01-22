import os
import threading
import time
from collections import defaultdict, deque
from kombu import Producer
from celery.app import app_or_default
from celery.utils.nodenames import anon_nodename
from celery.utils.time import utcoffset
from .event import Event, get_exchange, group_from
def _set_publisher(self, producer):
    self.producer = producer