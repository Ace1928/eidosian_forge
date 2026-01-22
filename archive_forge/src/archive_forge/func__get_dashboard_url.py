import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _get_dashboard_url(self) -> str:
    import ray.core.generated.ray_client_pb2 as ray_client_pb2
    return self.worker.get_cluster_info(ray_client_pb2.ClusterInfoType.DASHBOARD_URL).get('dashboard_url', '')