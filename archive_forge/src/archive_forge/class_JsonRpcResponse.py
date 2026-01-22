from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
@component_name(f'_Response')
class JsonRpcResponse(BaseModel):
    jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
    id: Union[StrictStr, int] = Field(None, example=0)
    result: dict

    class Config:
        extra = 'forbid'