import json
import logging
from pathlib import Path
from typing import Callable, Dict, Union
import yaml
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
def _load_prompt(config: dict) -> PromptTemplate:
    """Load the prompt template from config."""
    config = _load_template('template', config)
    config = _load_output_parser(config)
    template_format = config.get('template_format', 'f-string')
    if template_format == 'jinja2':
        raise ValueError(f"Loading templates with '{template_format}' format is no longer supported since it can lead to arbitrary code execution. Please migrate to using the 'f-string' template format, which does not suffer from this issue.")
    return PromptTemplate(**config)