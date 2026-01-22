from __future__ import annotations
import logging
from pymongo import monitoring
class ConnectionPoolLogger(monitoring.ConnectionPoolListener):
    """A simple listener that logs server connection pool events.

    Listens for :class:`~pymongo.monitoring.PoolCreatedEvent`,
    :class:`~pymongo.monitoring.PoolClearedEvent`,
    :class:`~pymongo.monitoring.PoolClosedEvent`,
    :~pymongo.monitoring.class:`ConnectionCreatedEvent`,
    :class:`~pymongo.monitoring.ConnectionReadyEvent`,
    :class:`~pymongo.monitoring.ConnectionClosedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutStartedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutFailedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckedOutEvent`,
    and :class:`~pymongo.monitoring.ConnectionCheckedInEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def pool_created(self, event: monitoring.PoolCreatedEvent) -> None:
        logging.info(f'[pool {event.address}] pool created')

    def pool_ready(self, event: monitoring.PoolReadyEvent) -> None:
        logging.info(f'[pool {event.address}] pool ready')

    def pool_cleared(self, event: monitoring.PoolClearedEvent) -> None:
        logging.info(f'[pool {event.address}] pool cleared')

    def pool_closed(self, event: monitoring.PoolClosedEvent) -> None:
        logging.info(f'[pool {event.address}] pool closed')

    def connection_created(self, event: monitoring.ConnectionCreatedEvent) -> None:
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection created')

    def connection_ready(self, event: monitoring.ConnectionReadyEvent) -> None:
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection setup succeeded')

    def connection_closed(self, event: monitoring.ConnectionClosedEvent) -> None:
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection closed, reason: "{event.reason}"')

    def connection_check_out_started(self, event: monitoring.ConnectionCheckOutStartedEvent) -> None:
        logging.info(f'[pool {event.address}] connection check out started')

    def connection_check_out_failed(self, event: monitoring.ConnectionCheckOutFailedEvent) -> None:
        logging.info(f'[pool {event.address}] connection check out failed, reason: {event.reason}')

    def connection_checked_out(self, event: monitoring.ConnectionCheckedOutEvent) -> None:
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection checked out of pool')

    def connection_checked_in(self, event: monitoring.ConnectionCheckedInEvent) -> None:
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection checked into pool')