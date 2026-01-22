import copy
import json
from typing import Any, Dict, List, Optional, Type, Union
import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs.chat_generation import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the element of the Json object."""
    key_name: str
    'The name of the key to return.'

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]