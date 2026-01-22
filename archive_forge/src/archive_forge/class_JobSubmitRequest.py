import asyncio
import json
import time
from dataclasses import dataclass, replace, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.packaging import parse_uri
from ray.experimental.internal_kv import (
from ray.util.annotations import PublicAPI
@dataclass
class JobSubmitRequest:
    entrypoint: str
    submission_id: Optional[str] = None
    job_id: Optional[str] = None
    runtime_env: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    entrypoint_num_cpus: Optional[Union[int, float]] = None
    entrypoint_num_gpus: Optional[Union[int, float]] = None
    entrypoint_memory: Optional[int] = None
    entrypoint_resources: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if not isinstance(self.entrypoint, str):
            raise TypeError(f'entrypoint must be a string, got {type(self.entrypoint)}')
        if self.submission_id is not None and (not isinstance(self.submission_id, str)):
            raise TypeError(f'submission_id must be a string if provided, got {type(self.submission_id)}')
        if self.job_id is not None and (not isinstance(self.job_id, str)):
            raise TypeError(f'job_id must be a string if provided, got {type(self.job_id)}')
        if self.runtime_env is not None:
            if not isinstance(self.runtime_env, dict):
                raise TypeError(f'runtime_env must be a dict, got {type(self.runtime_env)}')
            else:
                for k in self.runtime_env.keys():
                    if not isinstance(k, str):
                        raise TypeError(f'runtime_env keys must be strings, got {type(k)}')
        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(f'metadata must be a dict, got {type(self.metadata)}')
            else:
                for k in self.metadata.keys():
                    if not isinstance(k, str):
                        raise TypeError(f'metadata keys must be strings, got {type(k)}')
                for v in self.metadata.values():
                    if not isinstance(v, str):
                        raise TypeError(f'metadata values must be strings, got {type(v)}')
        if self.entrypoint_num_cpus is not None and (not isinstance(self.entrypoint_num_cpus, (int, float))):
            raise TypeError(f'entrypoint_num_cpus must be a number, got {type(self.entrypoint_num_cpus)}')
        if self.entrypoint_num_gpus is not None and (not isinstance(self.entrypoint_num_gpus, (int, float))):
            raise TypeError(f'entrypoint_num_gpus must be a number, got {type(self.entrypoint_num_gpus)}')
        if self.entrypoint_memory is not None and (not isinstance(self.entrypoint_memory, int)):
            raise TypeError(f'entrypoint_memory must be an integer, got {type(self.entrypoint_memory)}')
        if self.entrypoint_resources is not None:
            if not isinstance(self.entrypoint_resources, dict):
                raise TypeError(f'entrypoint_resources must be a dict, got {type(self.entrypoint_resources)}')
            else:
                for k in self.entrypoint_resources.keys():
                    if not isinstance(k, str):
                        raise TypeError(f'entrypoint_resources keys must be strings, got {type(k)}')
                for v in self.entrypoint_resources.values():
                    if not isinstance(v, (int, float)):
                        raise TypeError(f'entrypoint_resources values must be numbers, got {type(v)}')