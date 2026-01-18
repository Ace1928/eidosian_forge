import time
from collections import defaultdict
from typing import List
from ray._raylet import GcsClient
from ray.autoscaler.v2.schema import ClusterStatus, Stats
from ray.autoscaler.v2.utils import ClusterStatusParser
from ray.core.generated.autoscaler_pb2 import GetClusterStatusReply

    Get the cluster status from the autoscaler.

    Args:
        gcs_address: The GCS address to query.
        timeout: Timeout in seconds for the request to be timeout

    Returns:
        A ClusterStatus object.
    