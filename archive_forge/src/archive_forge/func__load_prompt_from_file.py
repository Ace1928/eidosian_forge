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
def _load_prompt_from_file(file: Union[str, Path]) -> BasePromptTemplate:
    """Load prompt from file."""
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    if file_path.suffix == '.json':
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f'Got unsupported file type {file_path.suffix}')
    return load_prompt_from_config(config)