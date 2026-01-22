from __future__ import annotations
from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from ..assistant_tool_param import AssistantToolParam
from ..file_search_tool_param import FileSearchToolParam
from ..code_interpreter_tool_param import CodeInterpreterToolParam
from ..assistant_tool_choice_option_param import AssistantToolChoiceOptionParam
from ..assistant_response_format_option_param import AssistantResponseFormatOptionParam
class AdditionalMessageAttachment(TypedDict, total=False):
    file_id: str
    'The ID of the file to attach to the message.'
    tools: Iterable[AdditionalMessageAttachmentTool]
    'The tools to add this file to.'