from typing_extensions import Literal
from ....._models import BaseModel
class MessageFile(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the message file was created.'
    message_id: str
    '\n    The ID of the [message](https://platform.openai.com/docs/api-reference/messages)\n    that the [File](https://platform.openai.com/docs/api-reference/files) is\n    attached to.\n    '
    object: Literal['thread.message.file']
    'The object type, which is always `thread.message.file`.'