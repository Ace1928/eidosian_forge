from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .file_search_tool_param import FileSearchToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
from .assistant_tool_choice_option_param import AssistantToolChoiceOptionParam
from .assistant_response_format_option_param import AssistantResponseFormatOptionParam
class ThreadMessageAttachment(TypedDict, total=False):
    file_id: str
    'The ID of the file to attach to the message.'
    tools: Iterable[ThreadMessageAttachmentTool]
    'The tools to add this file to.'