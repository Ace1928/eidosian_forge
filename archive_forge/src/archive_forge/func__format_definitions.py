import json
from typing import Dict, Iterator, List, Optional
from urllib.parse import quote
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _format_definitions(self, query: str, definitions: List[Dict]) -> str:
    formatted_definitions: List[str] = []
    for definition in definitions:
        formatted_definitions.extend(self._format_definition(definition))
    if len(formatted_definitions) == 1:
        return f"Definition of '{query}':\n{formatted_definitions[0]}"
    result = f"Definitions of '{query}':\n\n"
    for i, formatted_definition in enumerate(formatted_definitions, 1):
        result += f'{i}. {formatted_definition}\n'
    return result