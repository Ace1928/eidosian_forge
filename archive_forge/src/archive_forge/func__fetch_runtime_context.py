from typing import TYPE_CHECKING
from types import SimpleNamespace
def _fetch_runtime_context(self):
    import ray.core.generated.ray_client_pb2 as ray_client_pb2
    return self.worker.get_cluster_info(ray_client_pb2.ClusterInfoType.RUNTIME_CONTEXT)