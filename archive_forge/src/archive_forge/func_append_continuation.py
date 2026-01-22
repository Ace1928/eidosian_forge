import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def append_continuation(self, task_id: TaskID, continuation_task_id: TaskID) -> None:
    """Append continuation to a task."""
    continuation_root = self.continuation_root.get(task_id, task_id)
    self.prev_continuation[continuation_task_id] = task_id
    self.next_continuation[task_id] = continuation_task_id
    self.continuation_root[continuation_task_id] = continuation_root
    self.latest_continuation[continuation_root] = continuation_task_id