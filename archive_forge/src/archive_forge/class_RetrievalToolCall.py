from typing_extensions import Literal
from ....._models import BaseModel
class RetrievalToolCall(BaseModel):
    id: str
    'The ID of the tool call object.'
    retrieval: object
    'For now, this is always going to be an empty object.'
    type: Literal['retrieval']
    'The type of tool call.\n\n    This is always going to be `retrieval` for this type of tool call.\n    '