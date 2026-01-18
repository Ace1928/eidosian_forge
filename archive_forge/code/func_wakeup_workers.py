import time
from operator import itemgetter
from kombu import Queue
from kombu.connection import maybe_channel
from kombu.mixins import ConsumerMixin
from celery import uuid
from celery.app import app_or_default
from celery.utils.time import adjust_timestamp
from .event import get_exchange
def wakeup_workers(self, channel=None):
    self.app.control.broadcast('heartbeat', connection=self.connection, channel=channel)