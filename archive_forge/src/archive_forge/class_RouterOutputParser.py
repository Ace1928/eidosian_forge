from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, cast
from langchain_core.callbacks import (
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import root_validator
from langchain.chains import LLMChain
from langchain.chains.router.base import RouterChain
from langchain.output_parsers.json import parse_and_check_json_markdown
class RouterOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parser for output of router chain in the multi-prompt chain."""
    default_destination: str = 'DEFAULT'
    next_inputs_type: Type = str
    next_inputs_inner_key: str = 'input'

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            expected_keys = ['destination', 'next_inputs']
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if not isinstance(parsed['destination'], str):
                raise ValueError("Expected 'destination' to be a string.")
            if not isinstance(parsed['next_inputs'], self.next_inputs_type):
                raise ValueError(f"Expected 'next_inputs' to be {self.next_inputs_type}.")
            parsed['next_inputs'] = {self.next_inputs_inner_key: parsed['next_inputs']}
            if parsed['destination'].strip().lower() == self.default_destination.lower():
                parsed['destination'] = None
            else:
                parsed['destination'] = parsed['destination'].strip()
            return parsed
        except Exception as e:
            raise OutputParserException(f'Parsing text\n{text}\n raised following error:\n{e}')