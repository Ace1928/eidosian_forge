import logging
import re
from collections import defaultdict
from typing import List, Optional, Dict, AsyncIterable, Tuple, Callable
from ray.dashboard.modules.job.common import JOB_LOGS_PATH_TEMPLATE
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
from ray.util.state.state_manager import StateDataSourceClient
from ray._private.pydantic_compat import BaseModel
from ray.dashboard.datacenter import DataSource
def _verify_node_registered(self, node_id: str):
    if node_id not in self.client.get_all_registered_log_agent_ids():
        raise DataSourceUnavailable(f"Given node id {node_id} is not available. It's either the node is dead, or it is not registered. Use `ray list nodes` to see the node status. If the node is registered, it is highly likely a transient issue. Try again.")
    assert node_id is not None