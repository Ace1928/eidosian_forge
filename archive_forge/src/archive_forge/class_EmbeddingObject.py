from typing import List, Literal, Optional, Union
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class EmbeddingObject(ResponseModel):
    object: Literal['embedding'] = 'embedding'
    embedding: List[float]
    index: int