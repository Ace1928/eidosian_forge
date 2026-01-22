import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
class InstanceUpdatedSuscriber(metaclass=ABCMeta):
    """Subscribers to instance status changes."""

    @abstractmethod
    def notify(self, events: List[InstanceUpdateEvent]) -> None:
        pass