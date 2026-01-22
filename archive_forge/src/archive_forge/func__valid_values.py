from enum import Enum
from typing import Any, Dict, List, Type
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import root_validator
@property
def _valid_values(self) -> List[str]:
    return [e.value for e in self.enum]