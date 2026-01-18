import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
def wait_for_rendezvous_to_free(self, expected_version):
    """
        When there's an existing valid rendezvous in state 'final', we have to wait until the next opportunity to join.

        Such opportunity may come from:

        1. rendezvous state changed by someone else, in which case we unblock and retry.
        2. rendezvous becomes invalid because at least one member failed to renew their
           leased keep_alive node. We detect this, and destroy the rendezvous.
        """
    active_version, state = self.get_rdzv_state()
    while True:
        if state['status'] != 'final' or state['version'] != expected_version:
            return
        alive_members = self.client.get(self.get_path(f'/rdzv/v_{expected_version}'))
        keep_alive_keys = [ch.key for ch in alive_members.children]
        for key in state['keep_alives']:
            if key not in keep_alive_keys:
                log.info('Keep-alive key %s is not renewed.', key)
                log.info('Rendezvous version %s is incomplete. ', expected_version)
                log.info('Attempting to destroy it.')
                self.client.delete(key=self.get_path('/rdzv/active_version'), prevValue=active_version.value)
                log.info('Destroyed rendezvous version %s successfully.', expected_version)
                return
        try:
            overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
            self.client.watch(key=self.get_path('/rdzv'), index=active_version.etcd_index + 1, recursive=True, timeout=overall_timeout)
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass
        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError()
        active_version, state = self.get_rdzv_state()