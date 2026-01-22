from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
class Leaser(object):

    def __init__(self, manager: 'StreamingPullManager'):
        self._thread: Optional[threading.Thread] = None
        self._manager = manager
        self._operational_lock = threading.Lock()
        self._add_remove_lock = threading.Lock()
        self._leased_messages: Dict[str, _LeasedMessage] = {}
        self._bytes = 0
        'The total number of bytes consumed by leased messages.'
        self._stop_event = threading.Event()

    @property
    def message_count(self) -> int:
        """The number of leased messages."""
        return len(self._leased_messages)

    @property
    def ack_ids(self) -> KeysView[str]:
        """The ack IDs of all leased messages."""
        return self._leased_messages.keys()

    @property
    def bytes(self) -> int:
        """The total size, in bytes, of all leased messages."""
        return self._bytes

    def add(self, items: Iterable[requests.LeaseRequest]) -> None:
        """Add messages to be managed by the leaser."""
        with self._add_remove_lock:
            for item in items:
                if item.ack_id not in self._leased_messages:
                    self._leased_messages[item.ack_id] = _LeasedMessage(sent_time=float('inf'), size=item.byte_size, ordering_key=item.ordering_key)
                    self._bytes += item.byte_size
                else:
                    _LOGGER.debug('Message %s is already lease managed', item.ack_id)

    def start_lease_expiry_timer(self, ack_ids: Iterable[str]) -> None:
        """Start the lease expiry timer for `items`.

        Args:
            items: Sequence of ack-ids for which to start lease expiry timers.
        """
        with self._add_remove_lock:
            for ack_id in ack_ids:
                lease_info = self._leased_messages.get(ack_id)
                if lease_info:
                    self._leased_messages[ack_id] = lease_info._replace(sent_time=time.time())

    def remove(self, items: Iterable[Union[requests.AckRequest, requests.DropRequest, requests.NackRequest]]) -> None:
        """Remove messages from lease management."""
        with self._add_remove_lock:
            for item in items:
                if self._leased_messages.pop(item.ack_id, None) is not None:
                    self._bytes -= item.byte_size
                else:
                    _LOGGER.debug('Item %s was not managed.', item.ack_id)
            if self._bytes < 0:
                _LOGGER.debug('Bytes was unexpectedly negative: %d', self._bytes)
                self._bytes = 0

    def maintain_leases(self) -> None:
        """Maintain all of the leases being managed.

        This method modifies the ack deadline for all of the managed
        ack IDs, then waits for most of that time (but with jitter), and
        repeats.
        """
        while not self._stop_event.is_set():
            deadline = self._manager._obtain_ack_deadline(maybe_update=True)
            _LOGGER.debug('The current deadline value is %d seconds.', deadline)
            leased_messages = copy.copy(self._leased_messages)
            cutoff = time.time() - self._manager.flow_control.max_lease_duration
            to_drop = [requests.DropRequest(ack_id, item.size, item.ordering_key) for ack_id, item in leased_messages.items() if item.sent_time < cutoff]
            if to_drop:
                _LOGGER.warning('Dropping %s items because they were leased too long.', len(to_drop))
                assert self._manager.dispatcher is not None
                self._manager.dispatcher.drop(to_drop)
            for item in to_drop:
                leased_messages.pop(item.ack_id)
            ack_ids = leased_messages.keys()
            expired_ack_ids = set()
            if ack_ids:
                _LOGGER.debug('Renewing lease for %d ack IDs.', len(ack_ids))
                assert self._manager.dispatcher is not None
                ack_id_gen = (ack_id for ack_id in ack_ids)
                expired_ack_ids = self._manager._send_lease_modacks(ack_id_gen, deadline)
            start_time = time.time()
            if self._manager._exactly_once_delivery_enabled() and len(expired_ack_ids):
                assert self._manager.dispatcher is not None
                self._manager.dispatcher.drop([requests.DropRequest(ack_id, leased_messages.get(ack_id).size, leased_messages.get(ack_id).ordering_key) for ack_id in expired_ack_ids if ack_id in leased_messages])
            snooze = random.uniform(_MAX_BATCH_LATENCY, deadline * 0.9 - (time.time() - start_time))
            _LOGGER.debug('Snoozing lease management for %f seconds.', snooze)
            self._stop_event.wait(timeout=snooze)
        _LOGGER.debug('%s exiting.', _LEASE_WORKER_NAME)

    def start(self) -> None:
        with self._operational_lock:
            if self._thread is not None:
                raise ValueError('Leaser is already running.')
            self._stop_event.clear()
            thread = threading.Thread(name=_LEASE_WORKER_NAME, target=self.maintain_leases)
            thread.daemon = True
            thread.start()
            _LOGGER.debug('Started helper thread %s', thread.name)
            self._thread = thread

    def stop(self) -> None:
        with self._operational_lock:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()
            self._thread = None