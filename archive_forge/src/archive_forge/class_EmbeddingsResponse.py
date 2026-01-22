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
class EmbeddingsResponse(BaseModel):
    predictions: List[List[StrictFloat]]

    @validator('predictions', pre=True)
    def validate_predictions(cls, predictions):
        if isinstance(predictions, list) and (not predictions):
            raise ValueError('The input list is empty')
        if isinstance(predictions, list) and all((isinstance(item, list) and (not item) for item in predictions)):
            raise ValueError('One or more lists in the returned prediction response are empty')
        elif all((isinstance(item, float) for item in predictions)):
            return [predictions]
        else:
            return predictions