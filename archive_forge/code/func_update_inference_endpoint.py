from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
def update_inference_endpoint(self, name: str, *, accelerator: Optional[str]=None, instance_size: Optional[str]=None, instance_type: Optional[str]=None, min_replica: Optional[int]=None, max_replica: Optional[int]=None, repository: Optional[str]=None, framework: Optional[str]=None, revision: Optional[str]=None, task: Optional[str]=None, namespace: Optional[str]=None, token: Optional[str]=None) -> InferenceEndpoint:
    """Update an Inference Endpoint.

        This method allows the update of either the compute configuration, the deployed model, or both. All arguments are
        optional but at least one must be provided.

        For convenience, you can also update an Inference Endpoint using [`InferenceEndpoint.update`].

        Args:
            name (`str`):
                The name of the Inference Endpoint to update.

            accelerator (`str`, *optional*):
                The hardware accelerator to be used for inference (e.g. `"cpu"`).
            instance_size (`str`, *optional*):
                The size or type of the instance to be used for hosting the model (e.g. `"large"`).
            instance_type (`str`, *optional*):
                The cloud instance type where the Inference Endpoint will be deployed (e.g. `"c6i"`).
            min_replica (`int`, *optional*):
                The minimum number of replicas (instances) to keep running for the Inference Endpoint.
            max_replica (`int`, *optional*):
                The maximum number of replicas (instances) to scale to for the Inference Endpoint.

            repository (`str`, *optional*):
                The name of the model repository associated with the Inference Endpoint (e.g. `"gpt2"`).
            framework (`str`, *optional*):
                The machine learning framework used for the model (e.g. `"custom"`).
            revision (`str`, *optional*):
                The specific model revision to deploy on the Inference Endpoint (e.g. `"6c0e6080953db56375760c0471a8c5f2929baf11"`).
            task (`str`, *optional*):
                The task on which to deploy the model (e.g. `"text-classification"`).

            namespace (`str`, *optional*):
                The namespace where the Inference Endpoint will be updated. Defaults to the current user's namespace.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            [`InferenceEndpoint`]: information about the updated Inference Endpoint.
        """
    namespace = namespace or self._get_namespace(token=token)
    payload: Dict = {}
    if any((value is not None for value in (accelerator, instance_size, instance_type, min_replica, max_replica))):
        payload['compute'] = {'accelerator': accelerator, 'instanceSize': instance_size, 'instanceType': instance_type, 'scaling': {'maxReplica': max_replica, 'minReplica': min_replica}}
    if any((value is not None for value in (repository, framework, revision, task))):
        payload['model'] = {'framework': framework, 'repository': repository, 'revision': revision, 'task': task, 'image': {'huggingface': {}}}
    response = get_session().put(f'{INFERENCE_ENDPOINTS_ENDPOINT}/endpoint/{namespace}/{name}', headers=self._build_hf_headers(token=token), json=payload)
    hf_raise_for_status(response)
    return InferenceEndpoint.from_raw(response.json(), namespace=namespace, token=token)