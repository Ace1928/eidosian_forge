import copy
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Type
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.utils.json import parse_partial_json
class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse tools from OpenAI response."""
    strict: bool = False
    'Whether to allow non-JSON-compliant strings.\n\n    See: https://docs.python.org/3/library/json.html#encoders-and-decoders\n\n    Useful when the parsed output may include unicode characters or new lines.\n    '
    return_id: bool = False
    'Whether to return the tool call id.'
    first_tool_only: bool = False
    'Whether to return only the first tool call.\n\n    If False, the result will be a list of tool calls, or an empty list \n    if no tool calls are found.\n\n    If true, and multiple tool calls are found, only the first one will be returned,\n    and the other tool calls will be ignored. \n    If no tool calls are found, None will be returned. \n    '

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException('This output parser can only be used with a chat generation.')
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in tool_calls:
                if not self.return_id:
                    _ = tool_call.pop('id')
        else:
            try:
                raw_tool_calls = copy.deepcopy(message.additional_kwargs['tool_calls'])
            except KeyError:
                return []
            tool_calls = parse_tool_calls(raw_tool_calls, partial=partial, strict=self.strict, return_id=self.return_id)
        for tc in tool_calls:
            tc['type'] = tc.pop('name')
        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        return tool_calls

    def parse(self, text: str) -> Any:
        raise NotImplementedError()