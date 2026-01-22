from __future__ import annotations
from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from ..assistant_tool_param import AssistantToolParam
class RunCreateParamsBase(TypedDict, total=False):
    assistant_id: Required[str]
    '\n    The ID of the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) to use to\n    execute this run.\n    '
    additional_instructions: Optional[str]
    'Appends additional instructions at the end of the instructions for the run.\n\n    This is useful for modifying the behavior on a per-run basis without overriding\n    other instructions.\n    '
    instructions: Optional[str]
    '\n    Overrides the\n    [instructions](https://platform.openai.com/docs/api-reference/assistants/createAssistant)\n    of the assistant. This is useful for modifying the behavior on a per-run basis.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    model: Optional[str]
    '\n    The ID of the [Model](https://platform.openai.com/docs/api-reference/models) to\n    be used to execute this run. If a value is provided here, it will override the\n    model associated with the assistant. If not, the model associated with the\n    assistant will be used.\n    '
    tools: Optional[Iterable[AssistantToolParam]]
    'Override the tools the assistant can use for this run.\n\n    This is useful for modifying the behavior on a per-run basis.\n    '