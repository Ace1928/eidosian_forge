import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
def upsert_instance(self, instance: Instance, expected_instance_version: Optional[int]=None, expected_storage_verison: Optional[int]=None) -> StoreStatus:
    """Upsert an instance in the storage.
        If the expected_instance_version is specified, the update will fail
        if the current instance version does not match the expected version.
        Similarly, if the expected_storage_version is
        specified, the update will fail if the current storage version does not
        match the expected version.

        Note the version of the upserted instances will be set to the current
        storage version.

        Args:
            instance: The instance to be updated.
            expected_instance_version: The expected instance version.
            expected_storage_version: The expected storage version.

        Returns:
            StoreStatus: A tuple of (success, storage_version).
        """
    instance = copy.deepcopy(instance)
    instance.version = 0
    instance.timestamp_since_last_modified = int(time.time())
    result, version = self._storage.update(self._table_name, key=instance.instance_id, value=instance.SerializeToString(), expected_entry_version=expected_instance_version, expected_storage_version=expected_storage_verison, insert_only=False)
    if result:
        for subscriber in self._status_change_subscribers:
            subscriber.notify([InstanceUpdateEvent(instance_id=instance.instance_id, new_status=instance.status, new_ray_status=instance.ray_status)])
    return StoreStatus(result, version)