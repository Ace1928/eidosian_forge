from typing import List, Literal, Optional
from pydantic import Field
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class BaseRequestPayload(RequestModel):
    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[List[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = None