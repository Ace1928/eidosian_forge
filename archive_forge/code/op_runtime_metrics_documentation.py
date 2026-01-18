from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
Callback when a task is finished.