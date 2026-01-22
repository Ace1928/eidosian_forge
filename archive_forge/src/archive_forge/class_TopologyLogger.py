from __future__ import annotations
import logging
from pymongo import monitoring
class TopologyLogger(monitoring.TopologyListener):
    """A simple listener that logs server topology events.

    Listens for :class:`~pymongo.monitoring.TopologyOpenedEvent`,
    :class:`~pymongo.monitoring.TopologyDescriptionChangedEvent`,
    and :class:`~pymongo.monitoring.TopologyClosedEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def opened(self, event: monitoring.TopologyOpenedEvent) -> None:
        logging.info(f'Topology with id {event.topology_id} opened')

    def description_changed(self, event: monitoring.TopologyDescriptionChangedEvent) -> None:
        logging.info(f'Topology description updated for topology id {event.topology_id}')
        previous_topology_type = event.previous_description.topology_type
        new_topology_type = event.new_description.topology_type
        if new_topology_type != previous_topology_type:
            logging.info(f'Topology {event.topology_id} changed type from {event.previous_description.topology_type_name} to {event.new_description.topology_type_name}')
        if not event.new_description.has_writable_server():
            logging.warning('No writable servers available.')
        if not event.new_description.has_readable_server():
            logging.warning('No readable servers available.')

    def closed(self, event: monitoring.TopologyClosedEvent) -> None:
        logging.info(f'Topology with id {event.topology_id} closed')