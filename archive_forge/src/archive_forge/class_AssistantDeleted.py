from typing_extensions import Literal
from ..._models import BaseModel
class AssistantDeleted(BaseModel):
    id: str
    deleted: bool
    object: Literal['assistant.deleted']