import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
Return a concise string of cluster size to report to event logs.

        For example, "3 CPUs, 4 GPUs".
        