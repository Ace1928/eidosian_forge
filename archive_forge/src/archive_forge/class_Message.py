from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .message_content import MessageContent
class Message(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    assistant_id: Optional[str] = None
    '\n    If applicable, the ID of the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) that\n    authored this message.\n    '
    completed_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the message was completed.'
    content: List[MessageContent]
    'The content of the message in array of text and/or images.'
    created_at: int
    'The Unix timestamp (in seconds) for when the message was created.'
    file_ids: List[str]
    '\n    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs that\n    the assistant should use. Useful for tools like retrieval and code_interpreter\n    that can access files. A maximum of 10 files can be attached to a message.\n    '
    incomplete_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the message was marked as incomplete.'
    incomplete_details: Optional[IncompleteDetails] = None
    'On an incomplete message, details about why the message is incomplete.'
    metadata: Optional[object] = None
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    object: Literal['thread.message']
    'The object type, which is always `thread.message`.'
    role: Literal['user', 'assistant']
    'The entity that produced the message. One of `user` or `assistant`.'
    run_id: Optional[str] = None
    '\n    If applicable, the ID of the\n    [run](https://platform.openai.com/docs/api-reference/runs) associated with the\n    authoring of this message.\n    '
    status: Literal['in_progress', 'incomplete', 'completed']
    '\n    The status of the message, which can be either `in_progress`, `incomplete`, or\n    `completed`.\n    '
    thread_id: str
    '\n    The [thread](https://platform.openai.com/docs/api-reference/threads) ID that\n    this message belongs to.\n    '