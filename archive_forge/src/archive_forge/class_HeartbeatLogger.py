from __future__ import annotations
import logging
from pymongo import monitoring
class HeartbeatLogger(monitoring.ServerHeartbeatListener):
    """A simple listener that logs server heartbeat events.

    Listens for :class:`~pymongo.monitoring.ServerHeartbeatStartedEvent`,
    :class:`~pymongo.monitoring.ServerHeartbeatSucceededEvent`,
    and :class:`~pymongo.monitoring.ServerHeartbeatFailedEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def started(self, event: monitoring.ServerHeartbeatStartedEvent) -> None:
        logging.info(f'Heartbeat sent to server {event.connection_id}')

    def succeeded(self, event: monitoring.ServerHeartbeatSucceededEvent) -> None:
        logging.info(f'Heartbeat to server {event.connection_id} succeeded with reply {event.reply.document}')

    def failed(self, event: monitoring.ServerHeartbeatFailedEvent) -> None:
        logging.warning(f'Heartbeat to server {event.connection_id} failed with error {event.reply}')