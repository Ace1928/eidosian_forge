from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
def acquire_resources(self, resource_request: ResourceRequest) -> Optional[AcquiredResources]:
    if not self.has_resources_ready(resource_request):
        return None
    self._used_resources.append(resource_request)
    return self._resource_cls(bundles=resource_request.bundles, resource_request=resource_request)