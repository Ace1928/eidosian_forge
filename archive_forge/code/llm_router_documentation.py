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
Convenience constructor.