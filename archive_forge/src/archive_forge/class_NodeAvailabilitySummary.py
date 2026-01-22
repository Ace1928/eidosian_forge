import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
@dataclass
class NodeAvailabilitySummary:
    node_availabilities: Dict[str, NodeAvailabilityRecord]

    @classmethod
    def from_fields(cls, **fields) -> Optional['NodeAvailabilitySummary']:
        """Implement marshalling from nested fields. pydantic isn't a core dependency
        so we're implementing this by hand instead."""
        parsed = {}
        node_availabilites_dict = fields.get('node_availabilities', {})
        for node_type, node_availability_record_dict in node_availabilites_dict.items():
            unavailable_information_dict = node_availability_record_dict.pop('unavailable_node_information', None)
            unavaiable_information = None
            if unavailable_information_dict is not None:
                unavaiable_information = UnavailableNodeInformation(**unavailable_information_dict)
            parsed[node_type] = NodeAvailabilityRecord(unavailable_node_information=unavaiable_information, **node_availability_record_dict)
        return NodeAvailabilitySummary(node_availabilities=parsed)

    def __eq__(self, other: 'NodeAvailabilitySummary'):
        return self.node_availabilities == other.node_availabilities

    def __bool__(self) -> bool:
        return bool(self.node_availabilities)