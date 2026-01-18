import re
from typing import Callable, Dict, List
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, validator
Select which examples to use based on the input lengths.