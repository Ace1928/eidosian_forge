from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
The response may contain more than one game, so we need to choose the right
        one and return the id.