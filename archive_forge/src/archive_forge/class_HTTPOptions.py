import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI(stability='stable')
class HTTPOptions(BaseModel):
    """HTTP options for the proxies. Supported fields:

    - host: Host that the proxies listens for HTTP on. Defaults to
      "127.0.0.1". To expose Serve publicly, you probably want to set
      this to "0.0.0.0".
    - port: Port that the proxies listen for HTTP on. Defaults to 8000.
    - root_path: An optional root path to mount the serve application
      (for example, "/prefix"). All deployment routes are prefixed
      with this path.
    - request_timeout_s: End-to-end timeout for HTTP requests.
    - keep_alive_timeout_s: Duration to keep idle connections alive when no
      requests are ongoing.

    - location: [DEPRECATED: use `proxy_location` field instead] The deployment
      location of HTTP servers:

        - "HeadOnly": start one HTTP server on the head node. Serve
          assumes the head node is the node you executed serve.start
          on. This is the default.
        - "EveryNode": start one HTTP server per node.
        - "NoServer": disable HTTP server.

    - num_cpus: [DEPRECATED] The number of CPU cores to reserve for each
      internal Serve HTTP proxy actor.
    """
    host: Optional[str] = DEFAULT_HTTP_HOST
    port: int = DEFAULT_HTTP_PORT
    middlewares: List[Any] = []
    location: Optional[DeploymentMode] = DeploymentMode.HeadOnly
    num_cpus: int = 0
    root_url: str = ''
    root_path: str = ''
    request_timeout_s: Optional[float] = None
    keep_alive_timeout_s: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_S

    @validator('location', always=True)
    def location_backfill_no_server(cls, v, values):
        if values['host'] is None or v is None:
            return DeploymentMode.NoServer
        return v

    @validator('middlewares', always=True)
    def warn_for_middlewares(cls, v, values):
        if v:
            warnings.warn('Passing `middlewares` to HTTPOptions is deprecated and will be removed in a future version. Consider using the FastAPI integration to configure middlewares on your deployments: https://docs.ray.io/en/latest/serve/http-guide.html#fastapi-http-deployments')
        return v

    @validator('num_cpus', always=True)
    def warn_for_num_cpus(cls, v, values):
        if v:
            warnings.warn('Passing `num_cpus` to HTTPOptions is deprecated and will be removed in a future version.')
        return v

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True