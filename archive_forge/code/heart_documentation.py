from celery import bootsteps
from celery.worker import heartbeat
from .events import Events
Bootstep sending event heartbeats.

    This service sends a ``worker-heartbeat`` message every n seconds.

    Note:
        Not to be confused with AMQP protocol level heartbeats.
    