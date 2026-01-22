import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def _get_endpoint_for_route(self, route: str) -> Optional[EndpointTag]:
    for endpoint, info in self._endpoints.items():
        if info.route == route:
            return endpoint
    return None