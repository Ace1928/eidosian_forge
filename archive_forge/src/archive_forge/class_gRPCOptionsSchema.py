import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class gRPCOptionsSchema(BaseModel):
    """Options to start the gRPC Proxy with."""
    port: int = Field(default=DEFAULT_GRPC_PORT, description='Port for gRPC server. Defaults to 9000. Cannot be updated once Serve has started running. Serve must be shut down and restarted with the new port instead.')
    grpc_servicer_functions: List[str] = Field(default=[], description="List of import paths for gRPC `add_servicer_to_server` functions to add to Serve's gRPC proxy. Default to empty list, which means no gRPC methods will be added and no gRPC server will be started. The servicer functions need to be importable from the context of where Serve is running.")