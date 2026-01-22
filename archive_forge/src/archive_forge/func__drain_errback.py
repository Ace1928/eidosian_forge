import collections
import threading
import kombu
from kombu import exceptions as kombu_exceptions
from taskflow.engines.worker_based import dispatcher
from taskflow import logging
def _drain_errback(exc, interval):
    LOG.exception('Draining error: %s', exc)
    LOG.info('Retry triggering in %s seconds', interval)