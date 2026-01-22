from typing_extensions import Literal
from ....._models import BaseModel
class MessageCreation(BaseModel):
    message_id: str
    'The ID of the message that was created by this run step.'