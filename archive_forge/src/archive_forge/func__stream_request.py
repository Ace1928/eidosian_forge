import json
import time
from typing import Any, AsyncGenerator, AsyncIterable, Dict
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
def _stream_request(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    return send_stream_request(headers=self.auth_headers, base_url=self.base_url, path=path, payload=payload)