from typing import Dict, List
import numpy as np
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
def add_example(self, example: Dict[str, str]) -> None:
    """Add new example to list."""
    self.examples.append(example)