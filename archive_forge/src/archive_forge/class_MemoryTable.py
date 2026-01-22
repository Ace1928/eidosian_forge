import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
class MemoryTable:

    def __init__(self, entries: List[MemoryTableEntry], group_by_type: GroupByType=GroupByType.NODE_ADDRESS, sort_by_type: SortingType=SortingType.PID):
        self.table = entries
        self.group = {}
        self.summary = defaultdict(int)
        if group_by_type and sort_by_type:
            self.setup(group_by_type, sort_by_type)
        elif group_by_type:
            self._group_by(group_by_type)
        elif sort_by_type:
            self._sort_by(sort_by_type)

    def setup(self, group_by_type: GroupByType, sort_by_type: SortingType):
        """Setup memory table.

        This will sort entries first and group them after.
        Sort order will be still kept.
        """
        self._sort_by(sort_by_type)._group_by(group_by_type)
        for group_memory_table in self.group.values():
            group_memory_table.summarize()
        self.summarize()
        return self

    def insert_entry(self, entry: MemoryTableEntry):
        self.table.append(entry)

    def summarize(self):
        total_object_size = 0
        total_local_ref_count = 0
        total_pinned_in_memory = 0
        total_used_by_pending_task = 0
        total_captured_in_objects = 0
        total_actor_handles = 0
        for entry in self.table:
            if entry.object_size > 0:
                total_object_size += entry.object_size
            if entry.reference_type == ReferenceType.LOCAL_REFERENCE.value:
                total_local_ref_count += 1
            elif entry.reference_type == ReferenceType.PINNED_IN_MEMORY.value:
                total_pinned_in_memory += 1
            elif entry.reference_type == ReferenceType.USED_BY_PENDING_TASK.value:
                total_used_by_pending_task += 1
            elif entry.reference_type == ReferenceType.CAPTURED_IN_OBJECT.value:
                total_captured_in_objects += 1
            elif entry.reference_type == ReferenceType.ACTOR_HANDLE.value:
                total_actor_handles += 1
        self.summary = {'total_object_size': total_object_size, 'total_local_ref_count': total_local_ref_count, 'total_pinned_in_memory': total_pinned_in_memory, 'total_used_by_pending_task': total_used_by_pending_task, 'total_captured_in_objects': total_captured_in_objects, 'total_actor_handles': total_actor_handles}
        return self

    def _sort_by(self, sorting_type: SortingType):
        if sorting_type == SortingType.PID:
            self.table.sort(key=lambda entry: entry.pid)
        elif sorting_type == SortingType.OBJECT_SIZE:
            self.table.sort(key=lambda entry: entry.object_size)
        elif sorting_type == SortingType.REFERENCE_TYPE:
            self.table.sort(key=lambda entry: entry.reference_type)
        else:
            raise ValueError(f'Give sorting type: {sorting_type} is invalid.')
        return self

    def _group_by(self, group_by_type: GroupByType):
        """Group entries and summarize the result.

        NOTE: Each group is another MemoryTable.
        """
        self.group = {}
        group = defaultdict(list)
        for entry in self.table:
            group[entry.group_key(group_by_type)].append(entry)
        for group_key, entries in group.items():
            self.group[group_key] = MemoryTable(entries, group_by_type=None, sort_by_type=None)
        for group_key, group_memory_table in self.group.items():
            group_memory_table.summarize()
        return self

    def as_dict(self):
        return {'summary': self.summary, 'group': {group_key: {'entries': group_memory_table.get_entries(), 'summary': group_memory_table.summary} for group_key, group_memory_table in self.group.items()}}

    def get_entries(self) -> List[dict]:
        return [entry.as_dict() for entry in self.table]

    def __repr__(self):
        return str(self.as_dict())

    def __str__(self):
        return self.__repr__()