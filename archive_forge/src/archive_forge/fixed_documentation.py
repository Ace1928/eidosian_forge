from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
Fixed budget based resource manager.

    This resource manager keeps track of a fixed set of resources. When resources
    are acquired, they are subtracted from the budget. When resources are freed,
    they are added back to the budget.

    The resource manager still requires resources to be requested before they become
    available. However, because the resource requests are virtual, this will not
    trigger autoscaling.

    Additionally, resources are not reserved on request, only on acquisition. Thus,
    acquiring a resource can change the availability of other requests. Note that
    this behavior may be changed in future implementations.

    The fixed resource manager does not support placement strategies. Using
    ``STRICT_SPREAD`` will result in an error. ``STRICT_PACK`` will succeed only
    within a placement group bundle. All other placement group arguments will be
    ignored.

    Args:
        total_resources: Budget of resources to manage. Defaults to all available
            resources in the current task or all cluster resources (if outside a task).

    