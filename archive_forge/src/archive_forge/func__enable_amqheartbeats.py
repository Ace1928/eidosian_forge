import errno
import socket
from celery import bootsteps
from celery.exceptions import WorkerLostError
from celery.utils.log import get_logger
from . import state
def _enable_amqheartbeats(timer, connection, rate=2.0):
    heartbeat_error = [None]
    if not connection:
        return heartbeat_error
    heartbeat = connection.get_heartbeat_interval()
    if not (heartbeat and connection.supports_heartbeats):
        return heartbeat_error

    def tick(rate):
        try:
            connection.heartbeat_check(rate)
        except Exception as e:
            heartbeat_error[0] = e
    timer.call_repeatedly(heartbeat / rate, tick, (rate,))
    return heartbeat_error