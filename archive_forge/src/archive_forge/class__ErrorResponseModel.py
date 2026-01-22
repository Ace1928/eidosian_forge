from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@component_name(f'_ErrorResponse[{name}]', cls.__module__)
class _ErrorResponseModel(BaseModel):
    jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
    id: Union[StrictStr, int] = Field(None, example=0)
    error: _JsonRpcErrorModel

    class Config:
        extra = 'forbid'