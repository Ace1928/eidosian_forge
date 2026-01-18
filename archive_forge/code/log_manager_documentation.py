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
Categorize the given log files after filterieng them out using a given glob.

        Returns:
            Dictionary of {component_name -> list of log files}
        