from typing import List, Literal, Optional
from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
from mlflow.gateway.schemas.chat import BaseRequestPayload
class CompletionsUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None