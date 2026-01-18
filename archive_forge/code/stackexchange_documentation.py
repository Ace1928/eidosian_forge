import html
from typing import Any, Dict, Literal
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
Run query through StackExchange API and parse results.