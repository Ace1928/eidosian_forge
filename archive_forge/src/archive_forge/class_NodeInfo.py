import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class NodeInfo:
    instance_type_name: str
    ray_node_type_name: str
    instance_id: str
    ip_address: str
    node_status: Optional[str] = None
    node_id: Optional[str] = None
    resource_usage: Optional[NodeUsage] = None
    failure_detail: Optional[str] = None
    details: Optional[str] = None
    node_activity: Optional[List[str]] = None

    def total_resources(self) -> Dict[str, float]:
        if self.resource_usage is None:
            return {}
        return {r.resource_name: r.total for r in self.resource_usage.usage}

    def available_resources(self) -> Dict[str, float]:
        if self.resource_usage is None:
            return {}
        return {r.resource_name: r.total - r.used for r in self.resource_usage.usage}

    def used_resources(self) -> Dict[str, float]:
        if self.resource_usage is None:
            return {}
        return {r.resource_name: r.used for r in self.resource_usage.usage}