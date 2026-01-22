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
class EtcdRendezvousHandler(RendezvousHandler):
    """
    Implements a
    :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` interface
    backed by
    :py:class:`torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvous`.
    ``EtcdRendezvousHandler`` uses a URL to configure the type of rendezvous to
    use and to pass implementation specific configurations to the rendezvous
    module. The basic etcd rendezvous configuration URL looks like the following
    ::

     etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>  # noqa: W605

     -- example --

     etcd://localhost:2379/1234?min_workers=1&max_workers=3

    The URL above is interpreted as follows:

    1. Use the rendezvous handler that is registered with the ``etcd``
       scheme
    2. The ``etcd`` endpoint to use is ``localhost:2379``
    3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
       share a common etcd server for multiple jobs so long as the
       ``job_ids`` are guaranteed to be unique). Note that the job id can be
       any string (e.g. does not need to be a number) as long as it is
       unique.
    4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
       membership size - Torch Distributed Elastic starts running the job as
       long as the cluster size is greater than or equal to ``min_workers``
       and admits up to ``max_workers`` into the cluster.

    Below are a full list of the parameters that can be passed to etcd
    rendezvous:

    +--------------------------------------------+--------------------------+
    | Parameter                                  | Description              |
    +============================================+==========================+
    | min_workers                                | minimum number of        |
    |                                            | workers for the          |
    |                                            | rendezvous to be valid   |
    +--------------------------------------------+--------------------------+
    | max_workers                                | maximum number of        |
    |                                            | workers to admit         |
    +--------------------------------------------+--------------------------+
    | timeout                                    | total timeout within     |
    |                                            | which next_rendezvous is |
    |                                            | expected to succeed      |
    |                                            | (default 600s)           |
    +--------------------------------------------+--------------------------+
    | last_call_timeout                          | additional wait amount   |
    |                                            | (“last call”) after min  |
    |                                            | number of workers has    |
    |                                            | been reached (defaults   |
    |                                            | to 30s)                  |
    +--------------------------------------------+--------------------------+
    | etcd_prefix                                | path prefix (from etcd   |
    |                                            | root), inside which all  |
    |                                            | etcd nodes will be       |
    |                                            | created (defaults to     |
    |                                            | ``/torchelastic/p2p``)   |
    +--------------------------------------------+--------------------------+
    """

    def __init__(self, rdzv_impl):
        self._rdzv_impl = rdzv_impl

    def __del__(self):
        del self._rdzv_impl

    def get_backend(self) -> str:
        return 'etcd'

    def next_rendezvous(self):
        rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()
        log.info('Creating EtcdStore as the c10d::Store implementation')
        store = self._rdzv_impl.setup_kv_store(rdzv_version)
        return (store, rank, world_size)

    def is_closed(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            return state['status'] == 'closed'
        except etcd.EtcdKeyNotFound:
            return False

    def set_closed(self):
        self._rdzv_impl.set_closed()

    def num_nodes_waiting(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            if state['status'] == 'final':
                return state['num_workers_waiting']
        except etcd.EtcdKeyNotFound:
            pass
        return 0

    def get_run_id(self) -> str:
        return self._rdzv_impl._run_id

    def shutdown(self) -> bool:
        try:
            self.set_closed()
            return True
        except BaseException as e:
            log.warning('Shutdown failed. Error occurred: %s', str(e))
            return False