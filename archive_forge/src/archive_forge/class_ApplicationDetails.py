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
@PublicAPI(stability='stable')
class ApplicationDetails(BaseModel, extra=Extra.forbid, frozen=True):
    """Detailed info about a Serve application."""
    name: str = Field(description='Application name.')
    route_prefix: Optional[str] = Field(..., description='This is the `route_prefix` of the ingress deployment in the application. Requests to paths under this HTTP path prefix will be routed to this application. This value may be null if the application is deploying and app information has not yet fully propagated in the backend; or if the user explicitly set the prefix to `None`, so the application isn\'t exposed over HTTP. Routing is done based on longest-prefix match, so if you have deployment A with a prefix of "/a" and deployment B with a prefix of "/a/b", requests to "/a", "/a/", and "/a/c" go to A and requests to "/a/b", "/a/b/", and "/a/b/c" go to B. Routes must not end with a "/" unless they\'re the root (just "/"), which acts as a catch-all.')
    docs_path: Optional[str] = Field(..., description='The path at which the docs for this application is served, for instance the `docs_url` for FastAPI-integrated applications.')
    status: ApplicationStatus = Field(description='The current status of the application.')
    message: str = Field(description='A message that gives more insight into the application status.')
    last_deployed_time_s: float = Field(description='The time at which the application was deployed.')
    deployed_app_config: Optional[ServeApplicationSchema] = Field(description='The exact copy of the application config that was submitted to the cluster. This will include all of, and only, the options that were explicitly specified in the submitted config. Default values for unspecified options will not be displayed, and deployments that are part of the application but unlisted in the config will also not be displayed. Note that default values for unspecified options are applied to the cluster under the hood, and deployments that were unlisted will still be deployed. This config simply avoids cluttering with unspecified fields for readability.')
    deployments: Dict[str, DeploymentDetails] = Field(description='Details about the deployments in this application.')
    application_details_route_prefix_format = validator('route_prefix', allow_reuse=True)(_route_prefix_format)