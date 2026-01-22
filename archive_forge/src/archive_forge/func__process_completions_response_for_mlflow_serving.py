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
def _process_completions_response_for_mlflow_serving(response):
    try:
        validated_response = ServingTextResponse(**response)
        inference_data = validated_response.predictions
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return [completions.Choice(index=idx, text=entry, finish_reason=None) for idx, entry in enumerate(inference_data)]