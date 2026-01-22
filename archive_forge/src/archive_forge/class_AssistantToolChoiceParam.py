from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
from .assistant_tool_choice_function_param import AssistantToolChoiceFunctionParam
class AssistantToolChoiceParam(TypedDict, total=False):
    type: Required[Literal['function', 'code_interpreter', 'file_search']]
    'The type of the tool. If type is `function`, the function name must be set'
    function: AssistantToolChoiceFunctionParam