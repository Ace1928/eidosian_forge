import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
def add_status_change_subscriber(self, subscriber: InstanceUpdatedSuscriber):
    self._status_change_subscribers.append(subscriber)