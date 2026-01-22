import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
class ProxyRouter(ABC):
    """Router interface for the proxy to use."""

    @abstractmethod
    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]):
        raise NotImplementedError