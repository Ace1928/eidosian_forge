from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import root_validator
@root_validator()
def validate_parsers(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the parsers."""
    parsers = values['parsers']
    if len(parsers) < 2:
        raise ValueError('Must have at least two parsers')
    for parser in parsers:
        if parser._type == 'combining':
            raise ValueError('Cannot nest combining parsers')
        if parser._type == 'list':
            raise ValueError('Cannot combine list parsers')
    return values