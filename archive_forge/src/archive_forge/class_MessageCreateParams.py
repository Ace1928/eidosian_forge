from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal, Required, TypedDict
class MessageCreateParams(TypedDict, total=False):
    content: Required[str]
    'The content of the message.'
    role: Required[Literal['user']]
    'The role of the entity that is creating the message.\n\n    Currently only `user` is supported.\n    '
    file_ids: List[str]
    '\n    A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that\n    the message should use. There can be a maximum of 10 files attached to a\n    message. Useful for tools like `retrieval` and `code_interpreter` that can\n    access and use files.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '