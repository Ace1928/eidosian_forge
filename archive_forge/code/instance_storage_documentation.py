import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
Delete instances from the storage. If the expected_version is
        specified, the update will fail if the current storage version does not
        match the expected version.

        Args:
            to_delete: A list of instances to be deleted.
            expected_version: The expected storage version.

        Returns:
            StoreStatus: A tuple of (success, storage_version).
        