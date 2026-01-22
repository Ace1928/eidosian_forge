from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class ChatCompletionSystemMessageParam(TypedDict, total=False):
    content: Required[str]
    'The contents of the system message.'
    role: Required[Literal['system']]
    'The role of the messages author, in this case `system`.'
    name: str
    'An optional name for the participant.\n\n    Provides the model information to differentiate between participants of the same\n    role.\n    '