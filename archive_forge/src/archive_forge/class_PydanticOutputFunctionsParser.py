import copy
import json
from typing import Any, Dict, List, Optional, Type, Union
import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs.chat_generation import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a pydantic object."""
    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]
    'The pydantic schema to parse the output with.'

    @root_validator(pre=True)
    def validate_schema(cls, values: Dict) -> Dict:
        schema = values['pydantic_schema']
        if 'args_only' not in values:
            values['args_only'] = isinstance(schema, type) and issubclass(schema, BaseModel)
        elif values['args_only'] and isinstance(schema, Dict):
            raise ValueError('If multiple pydantic schemas are provided then args_only should be False.')
        return values

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        _result = super().parse_result(result)
        if self.args_only:
            pydantic_args = self.pydantic_schema.parse_raw(_result)
        else:
            fn_name = _result['name']
            _args = _result['arguments']
            pydantic_args = self.pydantic_schema[fn_name].parse_raw(_args)
        return pydantic_args