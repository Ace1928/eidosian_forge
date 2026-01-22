from typing import List, Literal, Optional
from pydantic import Field
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class ResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: Literal['chat.completion'] = 'chat.completion'
    created: int
    model: str
    choices: List[Choice]
    usage: ChatUsage

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA