import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
class MemoryTableEntry:

    def __init__(self, *, object_ref: dict, node_address: str, is_driver: bool, pid: int):
        self.is_driver = is_driver
        self.pid = pid
        self.node_address = node_address
        self.task_status = object_ref.get('taskStatus', '?')
        if self.task_status == 'NIL':
            self.task_status = '-'
        self.attempt_number = int(object_ref.get('attemptNumber', 0))
        if self.attempt_number > 0:
            self.task_status = f'Attempt #{self.attempt_number + 1}: {self.task_status}'
        self.object_size = int(object_ref.get('objectSize', -1))
        self.call_site = object_ref.get('callSite', '<Unknown>')
        if len(self.call_site) == 0:
            self.call_site = 'disabled'
        self.object_ref = ray.ObjectRef(decode_object_ref_if_needed(object_ref['objectId']))
        self.local_ref_count = int(object_ref.get('localRefCount', 0))
        self.pinned_in_memory = bool(object_ref.get('pinnedInMemory', False))
        self.submitted_task_ref_count = int(object_ref.get('submittedTaskRefCount', 0))
        self.contained_in_owned = [ray.ObjectRef(decode_object_ref_if_needed(object_ref)) for object_ref in object_ref.get('containedInOwned', [])]
        self.reference_type = self._get_reference_type()

    def is_valid(self) -> bool:
        if not self.pinned_in_memory and self.local_ref_count == 0 and (self.submitted_task_ref_count == 0) and (len(self.contained_in_owned) == 0):
            return False
        elif self.object_ref.is_nil():
            return False
        else:
            return True

    def group_key(self, group_by_type: GroupByType) -> str:
        if group_by_type == GroupByType.NODE_ADDRESS:
            return self.node_address
        elif group_by_type == GroupByType.STACK_TRACE:
            return self.call_site
        else:
            raise ValueError(f'group by type {group_by_type} is invalid.')

    def _get_reference_type(self) -> str:
        if self._is_object_ref_actor_handle():
            return ReferenceType.ACTOR_HANDLE.value
        if self.pinned_in_memory:
            return ReferenceType.PINNED_IN_MEMORY.value
        elif self.submitted_task_ref_count > 0:
            return ReferenceType.USED_BY_PENDING_TASK.value
        elif self.local_ref_count > 0:
            return ReferenceType.LOCAL_REFERENCE.value
        elif len(self.contained_in_owned) > 0:
            return ReferenceType.CAPTURED_IN_OBJECT.value
        else:
            return ReferenceType.UNKNOWN_STATUS.value

    def _is_object_ref_actor_handle(self) -> bool:
        object_ref_hex = self.object_ref.hex()
        taskid_random_bits_size = (TASKID_BYTES_SIZE - ACTORID_BYTES_SIZE) * 2
        actorid_random_bits_size = (ACTORID_BYTES_SIZE - JOBID_BYTES_SIZE) * 2
        random_bits = object_ref_hex[:taskid_random_bits_size]
        actor_random_bits = object_ref_hex[taskid_random_bits_size:taskid_random_bits_size + actorid_random_bits_size]
        if random_bits == 'f' * 16 and (not actor_random_bits == 'f' * 24):
            return True
        else:
            return False

    def as_dict(self):
        return {'object_ref': self.object_ref.hex(), 'pid': self.pid, 'node_ip_address': self.node_address, 'object_size': self.object_size, 'reference_type': self.reference_type, 'call_site': self.call_site, 'task_status': self.task_status, 'local_ref_count': self.local_ref_count, 'pinned_in_memory': self.pinned_in_memory, 'submitted_task_ref_count': self.submitted_task_ref_count, 'contained_in_owned': [object_ref.hex() for object_ref in self.contained_in_owned], 'type': 'Driver' if self.is_driver else 'Worker'}

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.as_dict())