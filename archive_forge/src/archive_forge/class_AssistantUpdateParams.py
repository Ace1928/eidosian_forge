from __future__ import annotations
from typing import List, Iterable, Optional
from typing_extensions import TypedDict
from .assistant_tool_param import AssistantToolParam
class AssistantUpdateParams(TypedDict, total=False):
    description: Optional[str]
    'The description of the assistant. The maximum length is 512 characters.'
    file_ids: List[str]
    '\n    A list of [File](https://platform.openai.com/docs/api-reference/files) IDs\n    attached to this assistant. There can be a maximum of 20 files attached to the\n    assistant. Files are ordered by their creation date in ascending order. If a\n    file was previously attached to the list but does not show up in the list, it\n    will be deleted from the assistant.\n    '
    instructions: Optional[str]
    'The system instructions that the assistant uses.\n\n    The maximum length is 32768 characters.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    model: str
    'ID of the model to use.\n\n    You can use the\n    [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n    see all of your available models, or see our\n    [Model overview](https://platform.openai.com/docs/models/overview) for\n    descriptions of them.\n    '
    name: Optional[str]
    'The name of the assistant. The maximum length is 256 characters.'
    tools: Iterable[AssistantToolParam]
    'A list of tool enabled on the assistant.\n\n    There can be a maximum of 128 tools per assistant. Tools can be of types\n    `code_interpreter`, `retrieval`, or `function`.\n    '