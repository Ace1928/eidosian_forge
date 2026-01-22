import time
from typing import List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError, validator
from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_SERVING_RESPONSE_KEY
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings
@staticmethod
def _process_payload(payload, key):
    payload = jsonable_encoder(payload, exclude_none=True)
    input_data = payload.pop(key, None)
    request_payload = {'inputs': input_data if isinstance(input_data, list) else [input_data]}
    if payload:
        request_payload['params'] = payload
    return request_payload