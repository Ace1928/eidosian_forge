from __future__ import annotations
from typing import Any, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers.format_instructions import (
def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(name=schema.name, description=schema.description, type=schema.type)