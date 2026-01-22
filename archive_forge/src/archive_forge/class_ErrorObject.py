from typing import Optional
from ..._models import BaseModel
class ErrorObject(BaseModel):
    code: Optional[str] = None
    message: str
    param: Optional[str] = None
    type: str