from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import root_validator
Parse the output of an LLM call.