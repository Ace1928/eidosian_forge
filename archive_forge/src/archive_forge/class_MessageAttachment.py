from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .file_search_tool_param import FileSearchToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
class MessageAttachment(TypedDict, total=False):
    file_id: str
    'The ID of the file to attach to the message.'
    tools: Iterable[MessageAttachmentTool]
    'The tools to add this file to.'