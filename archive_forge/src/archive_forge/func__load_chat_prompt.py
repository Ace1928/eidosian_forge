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
def _load_chat_prompt(config: Dict) -> ChatPromptTemplate:
    """Load chat prompt from config"""
    messages = config.pop('messages')
    template = messages[0]['prompt'].pop('template') if messages else None
    config.pop('input_variables')
    if not template:
        raise ValueError("Can't load chat prompt without template")
    return ChatPromptTemplate.from_template(template=template, **config)