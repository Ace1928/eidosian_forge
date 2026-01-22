import copy
import json
from typing import Any, Dict, List, Optional, Type, Union
import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs.chat_generation import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse an output as the Json object."""
    strict: bool = False
    'Whether to allow non-JSON-compliant strings.\n    \n    See: https://docs.python.org/3/library/json.html#encoders-and-decoders\n    \n    Useful when the parsed output may include unicode characters or new lines.\n    '
    args_only: bool = True
    'Whether to only return the arguments to the function call.'

    @property
    def _type(self) -> str:
        return 'json_functions'

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        if len(result) != 1:
            raise OutputParserException(f'Expected exactly one result, but got {len(result)}')
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException('This output parser can only be used with a chat generation.')
        message = generation.message
        if 'function_call' not in message.additional_kwargs:
            return None
        try:
            function_call = message.additional_kwargs['function_call']
        except KeyError as exc:
            if partial:
                return None
            else:
                raise OutputParserException(f'Could not parse function call: {exc}')
        try:
            if partial:
                if self.args_only:
                    return parse_partial_json(function_call['arguments'], strict=self.strict)
                else:
                    return {**function_call, 'arguments': parse_partial_json(function_call['arguments'], strict=self.strict)}
            elif self.args_only:
                try:
                    return json.loads(function_call['arguments'], strict=self.strict)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise OutputParserException(f'Could not parse function call data: {exc}')
            else:
                try:
                    return {**function_call, 'arguments': json.loads(function_call['arguments'], strict=self.strict)}
                except (json.JSONDecodeError, TypeError) as exc:
                    raise OutputParserException(f'Could not parse function call data: {exc}')
        except KeyError:
            return None

    def parse(self, text: str) -> Any:
        raise NotImplementedError()