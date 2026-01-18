from celery.signals import heartbeat_sent
from celery.utils.sysinfo import load_average
from .state import SOFTWARE_INFO, active_requests, all_total_count
Timer sending heartbeats at regular intervals.

    Arguments:
        timer (kombu.asynchronous.timer.Timer): Timer to use.
        eventer (celery.events.EventDispatcher): Event dispatcher
            to use.
        interval (float): Time in seconds between sending
            heartbeats.  Default is 2 seconds.
    