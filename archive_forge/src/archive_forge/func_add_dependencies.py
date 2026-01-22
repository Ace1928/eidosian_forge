import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def add_dependencies(self, task_id: TaskID, in_dependencies: List[TaskID]) -> None:
    """Add dependencies between a task and it input dependencies."""
    self.upstream_dependencies[task_id] = in_dependencies
    for in_task_id in in_dependencies:
        self.downstream_dependencies[in_task_id].append(task_id)