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
class LoggingConfig(BaseModel):
    """Logging config schema for configuring serve components logs.

    Example:

        .. code-block:: python

            from ray import serve
            from ray.serve.schema import LoggingConfig
            # Set log level for the deployment.
            @serve.deployment(LoggingConfig(log_level="DEBUG")
            class MyDeployment:
                def __call__(self) -> str:
                    return "Hello world!"
            # Set log directory for the deployment.
            @serve.deployment(LoggingConfig(logs_dir="/my_dir")
            class MyDeployment:
                def __call__(self) -> str:
                    return "Hello world!"
    """

    class Config:
        extra = Extra.forbid
    encoding: Union[str, EncodingType] = Field(default='TEXT', description="Encoding type for the serve logs. Default to 'TEXT'. 'JSON' is also supported to format all serve logs into json structure.")
    log_level: Union[int, str] = Field(default='INFO', description="Log level for the serve logs. Defaults to INFO. You can set it to 'DEBUG' to get more detailed debug logs.")
    logs_dir: Union[str, None] = Field(default=None, description="Directory to store the logs. Default to None, which means logs will be stored in the default directory ('/tmp/ray/session_latest/logs/serve/...').")
    enable_access_log: bool = Field(default=True, description='Whether to enable access logs for each request. Default to True.')

    @validator('encoding')
    def valid_encoding_format(cls, v):
        if v not in list(EncodingType):
            raise ValueError(f"Got '{v}' for encoding. Encoding must be one of {set(EncodingType)}.")
        return v

    @validator('log_level')
    def valid_log_level(cls, v):
        if isinstance(v, int):
            if v not in logging._levelToName:
                raise ValueError(f'Got "{v}" for log_level. log_level must be one of {list(logging._levelToName.keys())}.')
            return logging._levelToName[v]
        if v not in logging._nameToLevel:
            raise ValueError(f'Got "{v}" for log_level. log_level must be one of {list(logging._nameToLevel.keys())}.')
        return v

    def _compute_hash(self) -> int:
        return crc32((str(self.encoding) + str(self.log_level) + str(self.logs_dir) + str(self.enable_access_log)).encode('utf-8'))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LoggingConfig):
            return False
        return self._compute_hash() == other._compute_hash()