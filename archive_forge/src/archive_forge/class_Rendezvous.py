import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
class Rendezvous:
    """A rendezvous class for different actor/task processes to meet.

    To initialize an NCCL collective communication group, different
    actors/tasks spawned in Ray in a collective group needs to meet
    each other to synchronize the NCCLUniqueID. This class guarantees
    they meet via the NCCLUniqueIDStore, initialized on the rank=0
    process.

    Args:
        store_key: the unique store key, usually as a concatanation
            of group_name and communicator key. See `get_nccl_communicator`
            for more details.
    """

    def __init__(self, store_key):
        if not store_key:
            raise ValueError("Invalid store_key. The store_key is a concatenation of 'group_name' and the 'communicator_key'. See the docstring of `get_nccl_communicator` for details.")
        self._store_key = store_key
        self._store_name = None
        self._store = None

    def meet(self, timeout_s=180):
        """Meet at the named actor store.

        Args:
            timeout_s: timeout in seconds.

        Return:
            None
        """
        if timeout_s <= 0:
            raise ValueError("The 'timeout' argument must be positive. Got '{}'.".format(timeout_s))
        self._store_name = get_store_name(self._store_key)
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug("Trying to meet at the store '{}'".format(self._store_name))
                self._store = ray.get_actor(self._store_name)
            except ValueError:
                logger.debug("Failed to meet at the store '{}'.Trying again...".format(self._store_name))
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            logger.debug('Successful rendezvous!')
            break
        if not self._store:
            raise RuntimeError('Unable to meet other processes at the rendezvous store. If you are using P2P communication, please check if tensors are put in the correct GPU. ')

    @property
    def store(self):
        return self._store

    def get_nccl_id(self, timeout_s=180):
        """Get the NCCLUniqueID from the store through Ray.

        Args:
            timeout_s: timeout in seconds.

        Return:
            uid: the NCCLUniqueID if successful.
        """
        if not self._store:
            raise ValueError('Rendezvous store is not setup.')
        uid = None
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            uid = ray.get(self._store.get_id.remote())
            if not uid:
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            break
        if not uid:
            raise RuntimeError('Unable to get the NCCLUniqueID from the store.')
        return uid