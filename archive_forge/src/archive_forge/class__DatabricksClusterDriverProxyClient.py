import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
class _DatabricksClusterDriverProxyClient(_DatabricksClientBase):
    """An API client that talks to a Databricks cluster driver proxy app."""
    host: str
    cluster_id: str
    cluster_driver_port: str

    @root_validator(pre=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if 'api_url' not in values:
            host = values['host']
            cluster_id = values['cluster_id']
            port = values['cluster_driver_port']
            api_url = f'https://{host}/driver-proxy-api/o/0/{cluster_id}/{port}'
            values['api_url'] = api_url
        return values

    def post(self, request: Any, transform_output_fn: Optional[Callable[..., str]]=None) -> Any:
        resp = self._post(self.api_url, request)
        return transform_output_fn(resp) if transform_output_fn else resp